import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import defaultdict
from chemprop.distill.factory import RegisterDistill
from chemprop.nn_utils import get_activation_function

@RegisterDistill("base_distill")
class BaseDistill(nn.Module):
    def __init__(self, args):
        super(BaseDistill, self).__init__()
        self.accepts_multi_images = False
        self.args = args

        def create_moving_averages():
            return ExponentialMovingAverageMeter(args)

        self.exponential_average_meters = defaultdict(create_moving_averages)
        self.distill_loss = torch.FloatTensor(0)

    def forward(self, x):
        return x, {'student_z': x}

    def compute_loss(self, context):
        return self.distill_loss.to(context['device'])

    def additional_losses_to_log(self):
        return {'distill_loss': self.distill_loss.item()}

    def scale_loss(self, distill_loss):
        distill_avg = self.exponential_average_meters['distill_loss'].avg
        main_avg = self.exponential_average_meters['loss'].avg

        normalized_loss = distill_loss if distill_avg == 0 or main_avg == 0 else (distill_loss / distill_avg) * main_avg

        return normalized_loss * self.args.distill_lambda

    def update_meters(self, context):
        for key, loss in context.items():
            if key.endswith("loss"):
                self.exponential_average_meters[key].update(loss.item() if isinstance(loss, torch.Tensor) else loss)

@RegisterDistill("mse_distill")
class MseDistill(BaseDistill):
    def __init__(self, args):
        super(MseDistill, self).__init__(args)
        self.ffn = get_auxiliary_ffn(get_encoded_dim(args), args.target_features_size, args)
        self.mse = nn.MSELoss(reduction = 'mean')

    def forward(self, x, **kwargs):
        return x, {'student_z': self.ffn(x)}

    def mse_loss_fn(self, context):
        return self.mse(context['student_z'], context['target_features_batch'])

    def compute_loss(self, context):
        self.distill_loss = self.mse_loss_fn(context)
        return self.scale_loss(self.distill_loss)

@RegisterDistill("pred_as_hidden_mse_distill")
class PredAsHiddenMseDistill(MseDistill):
    def forward(self, x, **kwargs):
        z = self.ffn(x)
        return z.detach(), {'student_z': z}


@RegisterDistill("prediction_distill")
class PredictionDistill(BaseDistill):
    def __init__(self, args):
        super(PredictionDistill, self).__init__(args)
        self.ffn = get_auxiliary_ffn(args.target_features_size, args.output_size, args)

    def distill_loss_func(self, preds, targets):
        if self.args.dataset_type in ['classification', 'multiclass']:
            targets = torch.sigmoid(targets).unsqueeze(-1)
            targets = torch.cat((1 - targets, targets), dim=-1)
            preds = F.log_softmax(torch.cat((0 * preds.unsqueeze(-1), preds.unsqueeze(-1)), dim=-1), dim=-1)
            return F.kl_div(preds, targets, reduction='mean')
        else:
            return F.mse_loss(preds, targets, reduction='mean')

    def compute_loss(self, context):
        teacher_y = self.ffn(context['target_features_batch'])
        student_y = context['logits']
        teacher_loss = context['compute_loss_fn'](teacher_y)
        self.teacher_loss = teacher_loss.item()
        self.distill_loss = self.distill_loss_func(student_y, teacher_y.detach())
        return teacher_loss + self.scale_loss(self.distill_loss)


    def additional_losses_to_log(self):
        return {
            **super().additional_losses_to_log(),
            "teacher_loss": self.teacher_loss
        }



@RegisterDistill("regret_kl_distill")
class RegretKlDistill(BaseDistill):
    def __init__(self, args):
        super(RegretKlDistill, self).__init__(args)
        image_dim = args.target_features_size
        encoded_dim = get_encoded_dim(args)
        output_dim = args.output_size
        self.teacher_ffn = get_auxiliary_ffn(image_dim, output_dim, args)
        assert self.args.main_loss_lambda == 0

        self.heldout_student_ffn = get_auxiliary_ffn(encoded_dim, output_dim, args)
        self.heldout_teacher_ffn = get_auxiliary_ffn(image_dim, output_dim, args)

        self.heldout_student_ffn_frozen = get_auxiliary_ffn(encoded_dim, output_dim, args)
        self.heldout_teacher_ffn_frozen = get_auxiliary_ffn(image_dim, output_dim, args)

        self.main_student_ffn = get_auxiliary_ffn(encoded_dim, output_dim, args)
        self.main_teacher_ffn = get_auxiliary_ffn(image_dim, output_dim, args)

        self.gradient_reversal = GradientReversal()

    def distill_loss_func(self, preds, targets):
        if self.args.dataset_type in ['classification', 'multiclass']:
            return F.kl_div(preds, torch.sigmoid(targets), reduction='mean')
        else:
            return F.mse_loss(preds, targets, reduction='mean')

    def compute_loss(self, context):

        def compute_erm_loss():
            student_y = context['logits']
            heldout_student_y = context['heldout_logits']
            teacher_y = self.teacher_ffn(context['target_features_batch'])
            heldout_teacher_y = self.teacher_ffn(context['heldout_target_features_batch'])
            teacher_loss = context['compute_loss_fn'](teacher_y) + context['heldout_compute_loss_fn'](heldout_teacher_y)
            kl_loss = self.distill_loss_func(teacher_y.detach(), student_y) + self.distill_loss_func(heldout_teacher_y.detach(), heldout_student_y)
            erm_loss = context['loss'] + context['heldout_loss'] + teacher_loss

            return erm_loss, kl_loss

        def compute_heldout_loss():
            heldout_student_z = context['heldout_student_z'].detach()
            heldout_student_y = self.heldout_student_ffn(heldout_student_z)
            heldout_teacher_y = self.heldout_teacher_ffn(context['heldout_target_features_batch'])
            heldout_student_loss = context['heldout_compute_loss_fn'](heldout_student_y)
            heldout_teacher_loss = context['heldout_compute_loss_fn'](heldout_teacher_y)
            kl_loss = self.distill_loss_func(heldout_teacher_y.detach(), heldout_student_y)

            return heldout_student_loss + heldout_teacher_loss, kl_loss

        def compute_main_loss():
            raise NotImplementedError
            main_student_z = context['student_z']
            main_student_y = self.main_student_ffn(self.gradient_reversal(main_student_z))

            main_teacher_y = self.main_teacher_ffn(context['target_features_batch'])
            main_student_loss = context['compute_loss_fn'](main_student_y)
            main_teacher_loss = context['compute_loss_fn'](main_teacher_y)

            main_kl_loss = self.distill_loss_func(main_teacher_y.detach(), main_student_y)

            self.heldout_student_ffn_frozen.load_state_dict(self.heldout_student_ffn.state_dict())
            self.heldout_teacher_ffn_frozen.load_state_dict(self.heldout_teacher_ffn.state_dict())

            rgm_heldout_student_y = self.heldout_student_ffn_frozen(main_student_z)
            rgm_heldout_teacher_y = self.heldout_teacher_ffn_frozen(context['target_features_batch'])

            rgm_heldout_student_loss = context['compute_loss_fn'](rgm_heldout_student_y)
            rgm_heldout_teacher_loss = context['compute_loss_fn'](rgm_heldout_teacher_y)
            rgm_heldout_kl_loss = self.distill_loss_func(rgm_heldout_teacher_y.detach(), rgm_heldout_student_y)


            rgm_loss = rgm_heldout_student_loss + rgm_heldout_teacher_loss + self.args.distill_lambda * rgm_heldout_kl_loss

            return main_student_loss + main_teacher_loss, main_kl_loss + rgm_loss


        self.erm_loss = compute_erm_loss()
        self.heldout_loss = compute_heldout_loss()
        self.main_loss = compute_main_loss()

        self.distill_loss = self.erm_loss + self.heldout_loss + self.main_loss
        return self.distill_loss


    def additional_losses_to_log(self):
        return {
            **super().additional_losses_to_log(),
            "erm_loss": self.erm_loss,
            "heldout_loss": self.heldout_loss,
            "main_loss": self.main_loss,
        }


def get_encoded_dim(args):
    if args.features_only:
        first_linear_dim = args.features_size
    else:
        first_linear_dim = args.hidden_size
        if args.use_input_features:
            first_linear_dim += args.features_size

    return first_linear_dim


def get_auxiliary_ffn(first_linear_dim, output_dim, args):
    """
    Creates an FFN where input is x and output is z

    :param args: Arguments.
    """
    dropout = nn.Dropout(args.dropout)
    activation = get_activation_function(args.activation)

    # Create auxiliary FFN layers
    if args.distill_ffn_num_layers == 1:
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, output_dim)
        ]
    else:
        ffn = [
            dropout,
            nn.Linear(first_linear_dim, args.ffn_hidden_size)
        ]
        for _ in range(args.distill_ffn_num_layers - 2):
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
            ])
        ffn.extend([
            activation,
            dropout,
            nn.Linear(args.ffn_hidden_size, output_dim),
        ])

    # Create FFN model
    return nn.Sequential(*ffn)

class GradientReversal(nn.Module):
    def forward(self, x):
        return x

    def backward(self, grad_output):
        return -grad_output

# Average Meter
class ExponentialMovingAverageMeter:
    def __init__(self, args):
        self.reset()
        self.args = args

    def reset(self):
        self.val = 0
        self.avg = 0
        self.first_iter = True

    def update(self, val):
        self.val = val
        if isinstance(val, torch.Tensor):
            raise ValueError
        if self.first_iter:
            self.avg = val
            self.first_iter = False
        else:
            self.avg = (self.args.exponential_average_lambda) * self.avg + (1 - self.args.exponential_average_lambda) * self.val


