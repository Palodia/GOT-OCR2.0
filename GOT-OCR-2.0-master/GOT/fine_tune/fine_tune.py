import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from swift.llm import ModelType, InferArguments, infer_main, DatasetName, SftArguments, sft_main, merge_lora_main

def before_fine_tune_eval():
    # 微调前评估模型
    infer_args = InferArguments(
        model_type='got-ocr2',
        model_id_or_path='D:/Works/OCR/GOT-OCR/GOT/CR2.0/GOT-OCR-2.0-master',
        eval_human=True)
    infer_main(infer_args)


def fine_tune():
    # 微调模型
    # 创建 SftArguments 实例，配置微调参数
    sft_args = SftArguments(
                            model_type='got-ocr2',  # 指定模型类型为 got-ocr2
                            model_id_or_path='D:/Works/OCR/GOT-OCR/GOT/CR2.0/GOT-OCR-2.0-master',  # 指定模型的本地路径
                            dataset=[DatasetName.alpaca_zh, DatasetName.alpaca_en],  # 指定数据集，可以是多个数据集
                            train_dataset_sample=500,  # 指定训练数据集的样本数量
                            eval_steps=20,  # 每 20 步进行一次评估
                            logging_steps=5,  # 每 5 步记录一次日志
                            output_dir='output',  # 指定输出目录
                            lora_target_modules='ALL',  # 指定 LoRA 的目标模块，'ALL' 表示所有模块
                            self_cognition_sample=500,  # 指定自我认知样本数量
                            model_name=['', ''],  # 模型名称，可以为空
                            model_author=['', '']  # 模型作者，可以为空
                            )

    # 调用 sft_main 函数进行微调
    output = sft_main(sft_args)

    # 获取最佳模型的检查点路径
    best_model_checkpoint = output['best_model_checkpoint']

    # 打印最佳模型的检查点路径
    print(f'best_model_checkpoint: {best_model_checkpoint}')



def after_fine_tune_eval():
    # 微调后评估模型
    best_model_checkpoint = '/jppeng/gitapp/swift/output/qwen1half-7b-chat/v0-20240324-001719/checkpoint-62'
    infer_args = InferArguments(
        model_type='got-ocr2',
        model_id_or_path='D:/Works/OCR/GOT-OCR/GOT/CR2.0/GOT-OCR-2.0-master',
        ckpt_dir=best_model_checkpoint,
        eval_human=True)
    # merge_lora_main(infer_args)
    result = infer_main(infer_args)


if __name__ == "__main__":
    before_fine_tune_eval()
    fine_tune()
    after_fine_tune_eval()