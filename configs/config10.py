from benetech.metrics import BenetechMetric, AccuracyMetric
import imgaug.augmenters as iaa


args = {
    # glogal
    "fp16": True,
    "TRAIN_ANNOTATION_PATH": "data/df_images.csv",
    
    "device": "cuda",
    "save_path": "checkpoints",

    # training
    "epochs": 5,
    "batch_size": 2,
    "grad_accum_steps": 4, # effective batch size = batch_size * grad_accum_steps
    "grad_clip_norm": 1.0,
    "first_eval_epoch": 1,
    "validation_step": 20000,
    "max_patches": 1256,
    "max_token_len": 300,
    
    "extra_data": True,
    
    # model
    "backbone": "google/matcha-base",
    "freeze_encoder": False,
    "cls_head": True,
        
    # optimizer
    "learning_rate": 2.5e-5,
    "weight_decay": 1e-4,
    
    # lr scheduler
    "scheduler_warmup_epochs": 0,
    "max_learning_rate": 2.5e-5,
    "div_factor": 1.0,
    "final_div_factor": 10.0,
}

args["metrics_dict"] = {
    "BM": BenetechMetric(),
    "acc": AccuracyMetric()
}

args["train_transforms"] = iaa.Sequential([
        iaa.Sometimes(0.2, iaa.Affine(rotate=(-5, 5), fit_output=True)),
        iaa.Sometimes(0.1, iaa.Resize((0.8, 1.2))),
        iaa.Sometimes(0.35, iaa.SomeOf((1, 3), [ 
            iaa.AddToBrightness((-70, 70)),
            iaa.MultiplySaturation((0.4, 1.6)),
            iaa.MultiplyBrightness((0.4, 1.6)),
            iaa.MultiplyHue((0.6, 1.5)),
            iaa.RemoveSaturation(),
            iaa.AddToSaturation((-70, 70)),
            iaa.GammaContrast((0.4, 2.2)),
            iaa.LinearContrast((0.3, 1.7)),
            iaa.AddToHueAndSaturation((-70, 70), per_channel=True),
            iaa.AddToHue((-70, 70)),
            iaa.AddToSaturation((-70, 70)),
            iaa.Grayscale(alpha=(0.0, 1.0)),
            iaa.ChangeColorTemperature((1100, 10000)),
            iaa.Affine(scale={"x": (0.8, 1.2), "y": (0.8, 1.1)}),
            iaa.Affine(shear=(-6, 6), fit_output=True),
            iaa.ChangeColorspace(from_colorspace="RGB", to_colorspace="HSV"),
            iaa.GammaContrast((0.4, 2.2), per_channel=True),
            iaa.SigmoidContrast(gain=(2, 12), cutoff=(0.4, 0.6), per_channel=True),
            iaa.LogContrast(gain=(0.4, 1.6), per_channel=True)
        ]))
])