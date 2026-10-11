# LoRA base v2

本目录保存 Git `432eb14` 的完整任务双 LoRA v2 基准。相比 [v1](../lora_base_v1/)，本版显式使用 `epoch_shuffle` 无放回采样，分离模型、数据和增强随机种子，启用采样审计与精确数据位置恢复，并默认启用 W&B。模型和优化超参数仍沿用 1002，预算为 20k 步。

配置及操作说明分别见 [pi05_fr3_wuji_fulltask_c0_train.py](pi05_fr3_wuji_fulltask_c0_train.py) 和 [训练教程](训练教程_完整任务C0.md)。默认数据和归一化路径是占位示例，使用前必须填写实际值。
