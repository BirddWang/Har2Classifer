import SimpleITK as sitk
import numpy as np
import os

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

def preprocess(input_path, output_dir, mni_path):
    """
    Preprocess the input MRI image:
    1. Bias field correction
    2. Resample to 1mm isotropic
    3. Register to MNI space
    4. Crop/pad to (192, 224, 192)
    """
    os.makedirs(output_dir, exist_ok=True)

    # Load input image
    img = sitk.ReadImage(input_path, sitk.sitkFloat32)

    # Bias field correction
    img_n4 = bias_field_correction(img)
    sitk.WriteImage(img_n4, os.path.join(output_dir, "bias_corrected.nii.gz"))

    # Resample to isotropic spacing
    img_iso = resample_to_isotropic(img_n4)
    sitk.WriteImage(img_iso, os.path.join(output_dir, "resampled_iso.nii.gz"))

    # Load MNI template
    mni_img = sitk.ReadImage(mni_path, sitk.sitkFloat32)

    # Register to MNI space
    img_mni = register_to_mni(img_iso, mni_img)
    sitk.WriteImage(img_mni, os.path.join(output_dir, "registered_to_mni.nii.gz"))

    # Crop/pad to target shape
    img_final = crop_or_pad_to_shape(img_mni, (192, 224, 192))
    sitk.WriteImage(img_final, os.path.join(output_dir, "final_haca3_ready.nii.gz"))
    
    return img_final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess MRI images for HACA3.")
    parser.add_argument("input_path", type=str, default="data/Caltech/0051456/session_1/anat_1/mprage.nii.gz", help="Path to the input MRI image.")
    parser.add_argument("output_dir", type=str, default="data/preprocessed", help="Directory to save the preprocessed images.")
    parser.add_argument("mni_path", type=str, default="data/MNI152_T1_1mm.nii.gz", help="Path to the MNI template image.")
    args = parser.parse_args()

    preprocessed_img = preprocess(args.input_path, args.output_dir, args.mni_path)
    print(f"Preprocessed image saved to {os.path.join(args.output_dir, 'final_haca3_ready.nii.gz')}")
    print(f"Done!")
    exit(0)
