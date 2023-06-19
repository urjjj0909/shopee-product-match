# Shopee-product-match
蝦皮分類競賽[Shopee - Price Match Guarantee](https://www.kaggle.com/competitions/shopee-product-matching)希望參賽者能從商品圖片、標題、雜湊等資料判斷哪些是相同商品：

`posting_id` - the ID code for the posting.

`image` - the image id/md5sum.

`image_phash` - a perceptual hash of the image.

`title` - the product description for the posting.

`label_group` - ID code for all postings that map to the same product. Not provided for the test set.

## 11th place solution
本篇參考[第11名解題](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238181)：對`title`抽NLP Feature、對`image`抽CV Feature，最後組合二組特徵並分別定義閥值，超過這個閥值則判斷為相同商品。其中，在調整神經網路的技巧中使用了許多在優化器上的技術，以下分別針對各項做學習紀錄：

* ArcFace
* SAM（Sharpness-Aware Minimization）
* Gradual Warmup Learning Rate
* Ranger（RAdam + LookAhead）

## ArcFace
值得注意的是這場比賽並不是分類或迴歸問題，而是Feature的分群問題，如果只是用預訓練的模型去抽取Feature則沒有充分利用到資料集的標籤`label_group`。我們可以拔掉預訓練模型的輸出層、留下單純Feature extractor的部分再接上[ArcFace](https://www.kaggle.com/code/slawekbiel/arcface-explained)，基本上ArcFace是基於Cosine similarity的演算法並做以下步驟：

```
1. Normalize the embeddings and weights
2. Calculate the dot products
3. Calculate the angles with arccos
4. Add a constant factor m to the angle corresponding to the ground truth label
5. Turn angles back to cosines
6. Use cross entropy on the new cosine values to calculate loss
```

最主要在於第4點，我們會在`label`和抽出的`feature`間增加一個神經網路必須克服的角度差（Additive Angular Margin Loss，也是這篇論文的命名由來），使得越相似的資料在Feature space中能靠的越近。整體來說，就是利用資料的Ground truth去調整神經網路的權重，讓特徵向量具有較好的分群效果。

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/f2f25c6d-32cc-484b-ac9b-f9e8513cdb82" width="800"></div>

## SAM
SAM是一個[簡單且有效追求模型泛化（Generalization）能力的技巧](https://medium.com/ai-blog-tw/sharpness-aware-minimization-sam-%E7%B0%A1%E5%96%AE%E6%9C%89%E6%95%88%E5%9C%B0%E8%BF%BD%E6%B1%82%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B-257613bb365)
