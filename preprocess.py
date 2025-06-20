import SimpleITK as sitk
import numpy as np
import os, argparse, subprocess
from tqdm import tqdm

def convert_dicom_to_nifti(dicom_dir:str, output_dir:str, file_template:str="%n_%f_%d"):
    """
    Convert DICOM files to NIfTI format using dcm2niix.
    Args:
        dicom_dir (str): Directory containing DICOM files.
        output_dir (str): Directory to save the converted NIfTI files.
        file_template (str): Template for naming the output files.
    """
    try:
        subprocess.run([
            "dcm2niix_afni",
            "-z", "y", 
            "-o", output_dir,
            "-f", file_template,
            dicom_dir
        ], stdout=subprocess.DEVNULL)
    except Exception as e:
        print(f"Error: {e}")
        print(f"Error converting {dicom_dir} to NIfTI.")
        return False
    return True

# 1. Bias field correction using N4
def bias_field_correction(img):
    mask_img = sitk.OtsuThreshold(img, 0, 1, 200)
    corrector = sitk.N4BiasFieldCorrectionImageFilter()
    img_corrected = corrector.Execute(img, mask_img)
    return img_corrected

# 2. Resample to 1mm isotropic
def resample_to_isotropic(img, spacing=[1.0, 1.0, 1.0]):
    original_spacing = img.GetSpacing()
    original_size = img.GetSize()
    new_size = [
        int(round(osz * ospc / nspc)) for osz, ospc, nspc in zip(original_size, original_spacing, spacing)
    ]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(img.GetDirection())
    resampler.SetOutputOrigin(img.GetOrigin())
    resampler.SetInterpolator(sitk.sitkBSpline)
    return resampler.Execute(img)

# 3. Register to MNI using affine
def register_to_mni(moving, fixed):
    initial_transform = sitk.CenteredTransformInitializer(
        fixed, moving, sitk.AffineTransform(3),
        sitk.CenteredTransformInitializerFilter.GEOMETRY
    )
    registration_method = sitk.ImageRegistrationMethod()
    registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    registration_method.SetInterpolator(sitk.sitkLinear)
    registration_method.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=100)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    registration_method.SetInitialTransform(initial_transform, inPlace=False)
    registration_method.SetShrinkFactorsPerLevel([4, 2, 1])
    registration_method.SetSmoothingSigmasPerLevel([2, 1, 0])
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    final_transform = registration_method.Execute(fixed, moving)
    return sitk.Resample(moving, fixed, final_transform, sitk.sitkLinear, 0.0, moving.GetPixelID())

# 4. Center crop or pad to 192 x 224 x 192 (HACA3 requirement)
def crop_or_pad_to_shape(img, target_shape=(192, 224, 192)):
    arr = sitk.GetArrayFromImage(img)  # shape: [Z, Y, X]
    current_shape = arr.shape
    target_z, target_y, target_x = target_shape

    pad = [(0, 0)] * 3
    crop = [slice(None)] * 3
    new_arr = np.zeros(target_shape, dtype=arr.dtype)

    for i, (cur, tgt) in enumerate(zip(current_shape, (target_z, target_y, target_x))):
        if cur < tgt:
            pad_before = (tgt - cur) // 2
            pad_after = tgt - cur - pad_before
            pad[i] = (pad_before, pad_after)
        else:
            start = (cur - tgt) // 2
            crop[i] = slice(start, start + tgt)

    if any(p[0] > 0 or p[1] > 0 for p in pad):
        arr = np.pad(arr, pad, mode='constant')
    else:
        arr = arr[crop[0], crop[1], crop[2]]

    new_img = sitk.GetImageFromArray(arr)
    new_img.SetSpacing(img.GetSpacing())
    new_img.SetOrigin(img.GetOrigin())
    new_img.SetDirection(img.GetDirection())
    return new_img

def preprocess(input_path, mni_path, output_dir=""):
    """
    Preprocess the input MRI image:
    1. Bias field correction
    2. Resample to 1mm isotropic
    3. Register to MNI space
    4. Crop/pad to (192, 224, 192)
    """
    if output_dir == "":
        output_dir = os.path.dirname(input_path)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    name = os.path.basename(input_path).split(".")[0]

    # Load input image
    img = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Bias field correction
    img_n4 = bias_field_correction(img)
    # sitk.WriteImage(img_n4, os.path.join(output_dir, "bias_corrected.nii.gz"))

    # Resample to isotropic spacing
    img_iso = resample_to_isotropic(img_n4)
    # sitk.WriteImage(img_iso, os.path.join(output_dir, "resampled_iso.nii.gz"))

    # Load MNI template
    mni_img = sitk.ReadImage(mni_path, sitk.sitkFloat32)

    # Register to MNI space
    img_mni = register_to_mni(img_iso, mni_img)
    # sitk.WriteImage(img_mni, os.path.join(output_dir, "registered_to_mni.nii.gz"))

    # Crop/pad to target shape
    img_final = crop_or_pad_to_shape(img_mni, (192, 224, 192))
    sitk.WriteImage(img_final, os.path.join(output_dir, f"{name}_prep.nii.gz"))
    
    return img_final

def get_args():
    parser = argparse.ArgumentParser(description="Preprocess MRI images for HACA3.")
    parser.add_argument("--input_path", type=str, default="", help="Path to the input MRI image.")
    parser.add_argument("--mni_path", type=str, default="data/MNI152_T1_1mm.nii.gz", help="Path to the MNI template image.")
    parser.add_argument("--output_dir", type=str, default="/media/robin/4B48E5E39EAFFEB2/ADNI/AD-ALL-prep", help="Path to save the preprocessed images.")
    return parser.parse_args()

def get_files_path(input_path):
    """
    Get all file paths in the input directory.
    """
    input_paths = []
    for root, dirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".nii.gz"):
                input_paths.append(os.path.join(root, file))
    return input_paths

def main():
    args = get_args()
    paths = get_files_path("../haca3/data/nifti_output")
    input_paths = paths
    print(f"Total files found: {len(paths)}")
    # exist_name = get_exist_name()
    # input_paths = del_exist_file(paths, exist_name)[:1500]
    print(f"New paths to process: {len(input_paths)}")
    print(input_paths[:10])
    mni_path = args.mni_path
    print(f"Found {len(input_paths)} files to process.")

    for input_path in tqdm(input_paths):
        try:
            preprocess(input_path, mni_path, output_dir=args.output_dir)
        except Exception as e:
            print(f"Error processing {input_path}: {e}")
    print("Preprocessing completed.")

def get_exist_name():
    exist_name = []
    exist_dir = "/media/robin/4B48E5E39EAFFEB2/AD-to6-nii"
    for root, dirs, files in os.walk(exist_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                name = file.split("/")[-1].split(".")[0]
                exist_name.append(name)
    exist_dir = "/media/robin/4B48E5E39EAFFEB2/T1-AD-BASE-nii"
    for root, dirs, files in os.walk(exist_dir):
        for file in files:
            if file.endswith(".nii.gz"):
                name = file.split("/")[-1].split(".")[0]
                exist_name.append(name)
    
    print(f"Existing files: {len(exist_name)}")
    print(f"Existing names: {exist_name[:10]}...")
    return exist_name

def del_exist_file(input_paths, exist_name):
    """
    Delete files that already exist in the output directory.
    """
    new_paths = []
    for pth in input_paths:
        name = pth.split("/")[-1].split(".")[0]
        if name in exist_name:
            # print(f"File {name} already exists, skipping.")
            continue
        new_paths.append(pth)
    return new_paths



if __name__ == "__main__":
    main()