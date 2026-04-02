import os
import tensorflow as tf


class GPUManager:
    def __init__(self, gpu_ids="0", memory_growth=True):
        
        self.gpu_ids = gpu_ids
        self.memory_growth = memory_growth
        self._setup_env()
        import tensorflow as tf
        self.tf = tf
        self.strategy = self._setup_gpu()

    def _setup_env(self):
        os.environ['SM_FRAMEWORK'] = 'tf.keras'
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids

    def _setup_gpu(self):
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                if self.memory_growth:
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)

                print(f"GPU(s) configuré(s) : {gpus}")
                return tf.distribute.MirroredStrategy()
            except RuntimeError as e:
                print(f"Erreur GPU (déjà initialisé) : {e}")
        else:
            print("Aucun GPU détecté, passage en mode CPU.")

        return tf.distribute.get_strategy()

    def get_strategy(self):
        return self.strategy