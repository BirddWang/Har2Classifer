import subprocess
import torch.utils.tensorboard as tb

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


class Recorder:
    def __init__(self, log_dir='runs'):
        self.writer = tb.SummaryWriter(log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def log_lr(self, lr, step):
        self.writer.add_scalar('learning_rate', lr, step)
        self.writer.flush()

    def log_acc(self, acc, step):
        self.writer.add_scalar('train/accuracy', acc, step)
        self.writer.flush()

    def log_loss(self, loss, step):
        self.writer.add_scalar('train/loss', loss, step)
        self.writer.flush()

    def log_val_acc(self, val_acc, step):
        self.writer.add_scalar('val/accuracy', val_acc, step)
        self.writer.flush()
        
    def log_val_loss(self, val_loss, step):
        self.writer.add_scalar('val/loss', val_loss, step)
        self.writer.flush()

    def close(self):
        self.writer.close()