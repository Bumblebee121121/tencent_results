数据表：seq
路径：D:\AI-file\tecent\data\TencentGR-1M\seq
Parquet 文件数量：10

Schema：
user_id: int64
seq: list<element: struct<item_id: int64, action_type: int32, timestamp: int64> not null>
  child 0, element: struct<item_id: int64, action_type: int32, timestamp: int64> not null
      child 0, item_id: int64
      child 1, action_type: int32
      child 2, timestamp: int64
-- schema metadata --
org.apache.spark.version: '3.3.1'
org.apache.spark.sql.parquet.row.metadata: '{"type":"struct","fields":[{"' + 393

前两条样例：

样例 1：
{
  "user_id": 953188,
  "seq": {
    "length": 94,
    "preview": [
      {
        "item_id": 2905777,
        "action_type": 0,
        "timestamp": 1745919240
      },
      {
        "item_id": 728757,
        "action_type": 0,
        "timestamp": 1745919475
      }
    ]
  }
}

