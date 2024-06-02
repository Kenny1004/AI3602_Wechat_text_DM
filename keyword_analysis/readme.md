## Package Specifications

- `pyecharts==2.0.1`
- `jieba==0.42.1`
- `protobuf==4.25.1`
- `google==3.0.0`
- `protobuf==4.25.1`
- `lz4==4.3.2`

## Usage

This project provides the function of generating chat record word cloud.

For usage

1.put your message database `MSG.db`(which can be found in` .../WeChat Files/wxid_xxxxxxxx/Msg`) in `./data/MSG.db`

2.`python analysiss.py`

It will render a wordcloud in `./data/聊天统计/wordcloud.html`

For stopwords / newwords setting

Modify `./data/stopwords.txt` and `./data/new_words.txt`