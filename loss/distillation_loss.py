# loss/distillation_loss.py

import torch.nn.functional as F

def distillation_loss(student_logits, teacher_logits):
    return F.mse_loss(student_logits, teacher_logits)

