from mpa.registry import STAGES
from .inferrer import DetectionInferrer


@STAGES.register_module()
class DetectionEvaluator(DetectionInferrer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def run(self, model_cfg, model_ckpt, data_cfg, **kwargs):
        """Run evaluation stage for detection

        - Run inference
        - Run evaluation via MMDetection -> MMCV
        """
        self._init_logger()
        mode = kwargs.get('mode', 'train')
        if mode not in self.mode:
            return {}

        cfg = self.configure(model_cfg, model_ckpt, data_cfg, training=False, **kwargs)
        self.logger.info('evaluate!')

        # mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

        # Inference
        infer_results = super().infer(cfg)
        detections = infer_results['detections']

        # Evaluate inference results
        eval_kwargs = self.cfg.get('evaluation', {}).copy()
        for key in ['interval', 'tmpdir', 'start', 'gpu_collect', 'save_best']:
            eval_kwargs.pop(key, None)
        eval_result = self.dataset.evaluate(detections, **eval_kwargs)
        self.logger.info(eval_result)

        return dict(mAP=eval_result.get('bbox_mAP_50', 0.0))
