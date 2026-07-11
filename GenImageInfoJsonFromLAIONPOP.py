from multiprocessing import Pool
import polars as pl
import os
os.environ['HF_ENDPOINT'] = "https://hf-mirror.com"


# ['url', 'key', 'cogvlm_caption', 'llava_caption', 'nsfw_prediction', 'alt_txt', 'alt_txt_similarity', 'width', 'height', 'original_width', 'original_height', 'exif']

if __name__ == '__main__':
    datasetDF = pl.read_parquet(r'~/laion-pop/*.parquet')
    datasetDF = datasetDF.with_columns(HQ_CAP=pl.concat_list("cogvlm_caption", "llava_caption"),CAP=pl.concat_list("alt_txt"))
    imageInfoDF = pl.read_json(r'~/big_disk/laion-pop-dl/ImageInfo.json')
    imageInfoDF = imageInfoDF.with_columns(pl.col("IMG").str.strip_suffix(".heic").alias("key"))
    imageInfoDF = imageInfoDF.join(datasetDF.select(["key","HQ_CAP", "CAP"]),
                                   on="key",
                                   how="left",
                                   coalesce=True).drop("key")
    #print(imageInfoDF)
    imageInfoDF.write_json(r'~/big_disk/laion-pop-dl/ImageInfo.json')