import logging

import hydra
import t5
import tensorflow.compat.v1 as tf
from omegaconf import DictConfig, OmegaConf
import t5.models
from task_registry import register_datasets


TPU_TOPOLOGY = "v3-8"


def configure_runtime(tpu_name):
    tf_log = tf.get_logger()
    tf_log.removeHandler(tf_log.handlers[0])

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=tpu_name)
        tpu_address = tpu.get_master()
        logging.root.info("Running on TPU: %s", tpu_address)
    except ValueError:
        raise Exception(f"Failed to connect to {tpu_name} TPU")

    tf.enable_eager_execution()
    tf.config.experimental_connect_to_host(tpu_address)
    tf.disable_v2_behavior()

    return tpu_address


def run_training(tpu_address, config):
    model = t5.models.MtfModel(
        model_dir=config.model_dir,
        tpu=tpu_address,
        tpu_topology=TPU_TOPOLOGY,
        **config.model,
    )

    model.finetune(
        mixture_or_task_name=config.train_task,
        pretrained_model_dir=config.pretrained_dir,
        finetune_steps=config.finetune_steps,
    )


@hydra.main(config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    tpu_address = configure_runtime(cfg.tpu_name)
    register_datasets(cfg.datasets)
    run_training(tpu_address, cfg)


if __name__ == "__main__":
    main()
