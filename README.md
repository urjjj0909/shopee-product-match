# Shopee-product-match
蝦皮分類競賽[Shopee - Price Match Guarantee](https://www.kaggle.com/competitions/shopee-product-matching)希望參賽者能從商品圖片、標題、雜湊等資料判斷哪些是相同商品：

`posting_id` - the ID code for the posting.

`image` - the image id/md5sum.

`image_phash` - a perceptual hash of the image.

`title` - the product description for the posting.

`label_group` - ID code for all postings that map to the same product. Not provided for the test set.

## 11th place solution
本篇參考[第11名解題](https://www.kaggle.com/competitions/shopee-product-matching/discussion/238181)：對`title`抽NLP feature、對`image`抽CV feature，最後組合二組特徵並分別定義閥值，超過這個閥值則判斷為相同商品。其中，在調整神經網路上使用了許多優化技術，以下分別針對各項做學習紀錄：

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
SAM是一個[簡單且有效追求模型泛化（Generalization）能力的技巧](https://medium.com/ai-blog-tw/sharpness-aware-minimization-sam-%E7%B0%A1%E5%96%AE%E6%9C%89%E6%95%88%E5%9C%B0%E8%BF%BD%E6%B1%82%E6%A8%A1%E5%9E%8B%E6%B3%9B%E5%8C%96%E8%83%BD%E5%8A%9B-257613bb365)，在最小化`loss`時也同時最小化`sharpness`，讓模型收斂在較為平坦的而非尖銳（Sharp）的minima區域，假設我們具有充足的訓練資料，且訓練資料與測試資料分布相當接近但有微小差距，此時如果模型收斂於尖銳的minima區域，就容易在測試集上產生很大的誤差，可參考[下圖](https://openreview.net/pdf?id=H1oyRlYgg)：

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/538559ae-384a-4f18-8a56-26a81b4ce179" width="700"></div>
<br/>

SAM的核心想法是不僅追求損失函數最小化，同時考量損失函數曲面（Loss surface）的平坦程度：意指在某組神經網路權重下附近的`loss`都是差不多的，實際的更新參數步驟可參考[該篇論文](https://arxiv.org/abs/2010.01412)：

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/1b888bfa-2bbd-43c5-be58-820c81364ae0" width="700"></div>
<br/>

其中，在每一次的更新參數下須做二次反向傳播（Backpropagation）：第一次是計算參數`w`的梯度、第二次則是計算參數`w+ε`的梯度；另外，在使用上可以參考本篇[SAM Optimizer](https://github.com/davda54/sam)，直接在現有的優化器外層包SAM即可：

```
from sam import SAM
...

model = YourModel()
base_optimizer = torch.optim.SGD  # define an optimizer for the "sharpness-aware" update
optimizer = SAM(model.parameters(), base_optimizer, lr=0.1, momentum=0.9)
...

for input, output in data:

    # first forward-backward pass
    loss = loss_function(output, model(input))  # use this loss for any training statistics
    loss.backward()
    optimizer.first_step(zero_grad=True)
  
    # second forward-backward pass
    loss_function(output, model(input)).backward()  # make sure to do a full forward pass
    optimizer.second_step(zero_grad=True)
...
```

## Gradual Warmup Learning Rate
Pytorch中有許多[自定義學習率更新的方法](https://pytorch.org/docs/stable/optim.html)常見的有：

* `LambdaLR` – 根據自定義的`lambda`函式計算Learning rate
* `StepLR` – 每走固定的`step_size`後Learning rate會乘以自定義`gamma`做調整
* `ReduceLROnPlateau` – 當`metric`在自定義`patience`個Step內不再改善，Learning rate則乘以`factor`的速率下降

本篇參考[Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://github.com/ildoonet/pytorch-gradual-warmup-lr)架構將Pytorch提供的學習率更新方法用`GradualWarmupScheduler`包起來：

```
import torch
from torch.optim.lr_scheduler import StepLR, ExponentialLR
from torch.optim.sgd import SGD
from warmup_scheduler import GradualWarmupScheduler

if __name__ == '__main__':
    model = [torch.nn.Parameter(torch.randn(2, 2, requires_grad=True))]
    optim = SGD(model, 0.1)

    # scheduler_warmup is chained with schduler_steplr
    scheduler_steplr = StepLR(optim, step_size=10, gamma=0.1)
    scheduler_warmup = GradualWarmupScheduler(optim, multiplier=1, total_epoch=5, after_scheduler=scheduler_steplr)

    # this zero gradient update is needed to avoid a warning message, issue #8.
    optim.zero_grad()
    optim.step()

    for epoch in range(1, 20):
        scheduler_warmup.step(epoch)
        print(epoch, optim.param_groups[0]['lr'])
        optim.step()    # backward pass (update network)
```

## Ranger
Ranger其實是二個優化器`RAdam`和`LookAhead`的結合，首先對這二個優化器做簡單總結：

* RAdam可以看作是一個自動熱身（Warmup）版本的Adam
* LookAhead則是一個Loss surface的探索器，避免模型訓練時走錯路或落入Local minimum

### RAdam
Adam是一個很常使用到的自適應學習率（Adaptive learning rate）的優化器，但為人詬病的是訓練初期學習率的變化非常大，容易在模型只看少量訓練數據的訓練初期過度調整，導致容易收斂在Local minimum。為了解決這個問題，RAdam（Rectified Adam）根據Adaptive rate的變化（變異）程度來修正Learning rate，讓Adam也具有自動熱身的效果，也不需再手動調整參數。

參考下圖，X軸是Log scale的Gradient、Y軸是Frequency，把每個迭代的分布圖由上至下（訓練初期至訓練後期）堆疊在一起，可以觀察出左圖的Adam在訓練初期Gradient分布變化劇烈；但假設前幾個迭代的Learning rate很小，之後再自適應調整成長（Warmup），Gradient的分布就會是比較穩定的狀態，如右圖所示。

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/3a110a66-d332-4eee-b9ab-39f6b1d4254e" width="700"></div>
<br/>

作者在論文中利用估計去近似Gradient的變異程度，然而訓練初期估計無法準確，因此可以直接使用`SGDM`並且搭配較小的Learning rate（雖然走小步一點無法確保在Loss surface上往更好的方向走，但至少可以確定走小步一點不會讓Gradient差太多而導致訓練不穩定），而在訓練後期則可使用估計去修正Learning rate，那麼問題就很明顯有二個：

1. 如何確認初期和後期的切分點?
2. 如何對Learning rate做修正?

切分點主要是根據`ρ`來做計算，當看過的樣本數很少（訓練初期）直接使用`SGDM`來做模型參數更新、而看過的樣本數變多（訓練後期）則依據變異程度來修正Adaptive learning rate：

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/d6d80705-fb3e-4cea-966e-9391b202ebd4" width="700"></div>
<br/>

更詳細的數學式可直接參考該篇[論文](https://arxiv.org/abs/1908.03265)說明，參考[RAdam Oprimizer: Rectified Adam. 自動warmup版的Adam優化器](https://pedin024.medium.com/radam-oprimizer-rectified-adam-%E8%87%AA%E5%8B%95warmup%E7%89%88%E7%9A%84adam%E5%84%AA%E5%8C%96%E5%99%A8-ac9de9938a7f)總結`RAdam`具有以下優點：

1. RAdam根據Adaptive rate的變異程度去修正Learning rate，讓Adam也具有自動熱身的效果，也不需再手動調整參數
2. 享有Adam快速收斂優勢的同時，又達到跟SGD差不多好的收斂結果
3. 增加訓練的穩定性：看下圖可知，用不同的Learning rate對於RAdam最後的收斂結果沒有差很多，反觀Adam和SGD對於Learning rate設定比較敏感

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/55a750d0-e7ad-48cd-90c0-d3acf09de400" width="700"></div>

### LookAhead
LookAhead可以任意選擇搭配的優化器，先快速的探索更新`fast`的權重 (Inner update)，並在每k個mini-batch時，利用`fast`的權重更新`slow`的權重，再用`slow`的權重回去更新`fast`的權重（outer update），如此為一個循環，讓‵fast‵不容易走錯路或落入Local minimumm。

下圖左是準確率的等高線圖，可以看到當`fast`將探索的權重更新至`slow`權重，除了讓準確率往更高的方向前進外，也同時因為`slow`權重的更新拉動`fast`權重，讓它可以往準確率更高的區域繼續做探索：

<div align=center><img src="https://github.com/urjjj0909/shopee-product-match/assets/100120881/55c285c2-4b0e-4cf7-b746-b7f193652b37" width="700"></div>

參考[深度學習優化器Ranger: a synergistic optimizer using RAdam (Rectified Adam), Gradient Centralization and LookAhead筆記]([https://pedin024.medium.com/radam-oprimizer-rectified-adam-%E8%87%AA%E5%8B%95warmup%E7%89%88%E7%9A%84adam%E5%84%AA%E5%8C%96%E5%99%A8-ac9de9938a7f](https://pedin024.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E5%84%AA%E5%8C%96%E5%99%A8ranger-a-synergistic-optimizer-using-radam-rectified-adam-gradient-centralization-and-f022d9dd4217)https://pedin024.medium.com/%E6%B7%B1%E5%BA%A6%E5%AD%B8%E7%BF%92%E5%84%AA%E5%8C%96%E5%99%A8ranger-a-synergistic-optimizer-using-radam-rectified-adam-gradient-centralization-and-f022d9dd4217)總結`LookAhead`具有以下優點：

1. 增加訓練的穩定性
2. 減少調整超參數的工作量：就算是用預設設定，也能顯著的改善SGD和Adam的結果
3. 更快的收斂速度、更小的計算開銷
