import os
import sys
from pathlib import Path

import loguru
from echo_logger import monit_feishu
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent))
from _config import pdir

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # 根据你的显卡情况设置。如果显卡是0号卡，那么这里设置为'0'

from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


def bin_(_label):
    if _label is not None and type(_label) == str and 'positive' in _label:
        return 1
    return 0


@monit_feishu(title_ok='emotion_classification')
def main():
    model = AutoModelForSequenceClassification.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
    tokenizer = AutoTokenizer.from_pretrained('uer/roberta-base-finetuned-jd-binary-chinese')
    # use cuda
    loguru.logger.info(f"Loaded model")
    model = model.to('cuda')

    text_classification = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

    # csv_file = Path(pdir) / 'datasets' / 'dataset_emotion.csv.txt'
    csv_file = Path(pdir) / 'datasets' / 'wechat_data.csv.txt'
    import pandas as pd
    df = pd.read_csv(csv_file)

    # Check if 'label' and 'score' columns exist, otherwise classify and add 'label' column
    if 'label' not in df.columns:
        tqdm.pandas()

        def classify_text(text):
            text = str(text)
            try:
                return bin_(text_classification(text)[0]['label'])
            except RuntimeError as e:
                loguru.logger.warning(f"Skipping text due to error: {e}. Original Text: \n{text}")
                return bin_(None)

        # Classify text emotion with progress bar
        df['label'] = df['StrContent'].progress_apply(classify_text)

        # Save the updated dataframe to the same file
        df.to_csv(csv_file, index=False)

    # Convert date to datetime format
    df['Date'] = pd.to_datetime(df['StrTime'], format='%Y/%m/%d %H:%M')

    # Filter users
    # users = ['Tiān tiān', 'Serein']
    users = ['平凡', 'Kenny']
    # user_emotions = df[df['NickName'].isin(users)]
    user_emotions = df[df['Remark'].isin(users)]

    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    # Plot the results for each user
    plt.figure(figsize=(30, 10))
    for user in users:
        # user_daily_emotions = user_emotions[user_emotions['NickName'] == user].groupby(
        user_daily_emotions = user_emotions[user_emotions['Remark'] == user].groupby(
            user_emotions['Date'].dt.date).apply(lambda x: (x['label'] == 1).sum() / len(x)).reset_index(
            name='Positive Rate')
        plt.plot(user_daily_emotions['Date'], user_daily_emotions['Positive Rate'], marker='o', label=user)
    plt.rcParams['font.family'] = 'simhei'
    # Formatting the plot
    plt.title('Positive Rate Over Time by User')
    plt.xlabel('Date')
    plt.ylabel('Positive Rate')
    plt.legend(title='User')
    plt.grid(True)
    # Set x-axis to display date labels at a daily interval
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
    plt.xticks(rotation=45)
    plt.tight_layout()
    # Save as svg and png

    plt.savefig('emotion_classification_rate.svg')
    plt.savefig('emotion_classification_rate.png')
    plt.show()


if __name__ == '__main__':
    main()
