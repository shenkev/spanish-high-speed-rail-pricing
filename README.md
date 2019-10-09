# spanish-high-speed-rail-pricing

## Hyperparameter Search Results

### First Sweep (1% data)

- num_trees - Not clear.
- lr - Big learning rate leads to divergence, small learning rate leads to bad performance. Optimal lr has some tolerance (about 0.1 to 1.5 so 15x range).
- depth - Weak signal that deeper is better. Depth 8 works but depth of 50 can still work well, in fact the best models are 35, 45, 47. Need to investigate even deeper models.
- colsample_bytree - No clear signal.
- L1 Reg - No clear signal. Weak signal for if it's less than 1 then it's okay.
- L2 Reg - No clear signal.
- subsample - Weak signal that higher is better (>0.8).

### Next Steps

- num_trees - let's push this to see when it breaks (2-500)
- lr - settle on ~0.25
- depth - try deeper (2-500)
- colsample_bytree - don't bother (can try as regularization later)
- subsample - don't bother (can try as regularization later)
- L1 Reg - set to 1 (can try as regularization later)
- L2 Reg - set to 1 (can try as regularization later)

### Second Sweep (1% data)

I swept depth and number of trees from 2 to 500.

- High depth and low trees equal underfitting (why?). Basically the model doesn't work. Need > 6 trees to perform reasonably.
- The sweet spot is a trade-off between trees and depth. Depth 6-25 work well. Trees is flexible, 30-400 all work. For smaller depth numbers (6-10), you need higher tree numbers (200+) as opposed to (~50) if tree is 15-25 depth.
- More trees doesn't actually overfit, it just slows the algorithm down (linearly).
- Very deep (464 depth) can work but it's rare (you need to get lucky). 
- 200 depth works decently but it's not as good as lower numbers like 15.
- About 50 tree is good.
- Runs are pretty consistent across random seeds for the same depth and trees.

Conclusion
1. use around 15 +/- 7 depth
2. use around 50 +/- 20 trees
3. up to 400 trees doesn't actually overfit, it just slows the algorithm down (linearly)
4. up to 200 depth works but lower numbers like 15 is better and much faster

### Next Steps

- num_trees - (30-60)
- depth - (8-20)

### Third Sweep (10% data)

- no difference within these hyperparameter ranges
- loss falls from 0.417 to 0.242 RMSE using 10% data
- no overfitting occuring (likely because I split  train and val randomly so they're from the same distribution)

### Fourth Sweep (100% data)

- performance plateaus at 0.243 which is surprising

### Fifth Sweep (0.1% data)

I'm going to try to make the model overfit by using little training data and splitting train and val in time.

- With 8k train 2k val data (val is now out of time), there's overfitting
- 0.42 train rmse, 0.635 train rmse
- having more rounds helps a bit now (~50)
- having a shallow tree helps now (~9)
- the weird thing is I can't push the train rmse any lower by adding more trees and rounds


Conclusion

1. When we're OOT validatin and prone to overfitting, using shallower trees and more rounds helps (at least in low data regime).
2. Train and val error seem to be closely tied (more so than neural network models), I can't seem to cause overfitting.


### 6th Sweep (0.1% data)

I'm going to try regularization techniques. They seem to work.

Winning formula: 0.42 val rsme

'alpha':3.7107730190809587
'colsample_bynode':0.6796916592196786
'eta':0.20565322153244497
'eval_metric':'rmse'
'lambda':2.3335571359982987
'max_depth':28
'rounds':84
'subsample':0.8512193177339286

Runner ups:

'alpha':13.153248896628392
'colsample_bynode':0.6681831062420552
'eta':0.13561428796139682
'eval_metric':'rmse'
'lambda':5.847788248580009
'max_depth':19
'rounds':95
'subsample':0.5319581036595756

'alpha':4.501180104767641
'colsample_bynode':0.7798523533049491
'eta':0.06645071667404864
'eval_metric':'rmse'
'lambda':23.314379542828636
'max_depth':20
'rounds':81
'subsample':0.7495687981780765

These solutions have the following in common:

- large number of trees (~80, up from 30 we needed before)
- large combination of L2 and L1 reg
- (~0.7) colsample_bynode
- slightly more depth (~25, up from 15 we needed before)
- subsample (~0.75)

Conclusions

1. To reduce overfitting, increase number of trees and depth (approximately double both) then apply regularization.
2. Regularization should be about 0.7 colsample_bynode, 0.75 subsample, more L2 and L1

## 7th Sweep (0.001% data)

I now want to sweep hyperparameters individually to see if they really have an effect on the winning solution.

I'm starting off with the parameters:

- trees - 80
- depth - 25
- alpha - 5
- lambda - 5
- colsample_bynode - 0.8
- eta - 0.1
- subsample - 0.75

Sweeping lr (0.003 to 3)

- (~0.1) works the best
- < 0.05 and > 0.2 is starting to work worse
- before 0.25 lr worked well and there was a much better window of tolerance

Sweeping colsample_bynode (0.25 to 1)

- anything high or low works worse (0.455)
- anything in the range 0.5-0.8 works well (0.421)

Sweeping subsample (0.25 to 1)

- best is 0.7-0.8 (0.42)
- 0.4-0.9 is reasonable (0.428)
- anything lower or higher does slightly worse (0.439)

Sweeping alpha (1 to 125)

- 100 definitely doesn't work
- 3 works, 20 works (seems to have a decently large range)
- at 0.01 we get slightly worse performance (0.436)
- L1 by itself cannot achieve 0.42 performance (L1 is not as important as L2)

Sweeping lambda (1 to 25)

- using just L2 reg, we need (~20) to achieve 0.42 performance but using both L1 and L2, L2 only needs to be 5
- L2 can work by itself, you don't need L1 but L1 seems to make it easier to find good L2 values
- (20-40) achieves the best performance (0.42)
- 0.1 or lower does worse (0.449)
- 40+ does a bit worse

## 7th Sweep (0.001% data)

I'm going to try DART booster.

- trees - 80
- depth - 25
- alpha - 5
- lambda - 20
- colsample_bynode - 0.65
- eta - 0.1
- subsample - 0.8

- Dart seems to run faster?
- Dart is working a lot worse (0.482) out of the box even if I sets dropout rate to 0, why?
- Performance decreases monotonically with dropout rate (at 0.3 dropout, we see bad underfitting).
- I thought it was because L2 and L1 reg + Dart is constraining capacity too much. But even turning off L2 and L1 reg, Dart doesn't work.


Conclusions

1. Ranked list of most important parameters to avoid overfitting: lr, L2 reg, colsample_bynode, subsample, L1 reg
2. You also need to reduce lr to prevent overfitting (about 2.5x)
3. Range of viable lr decreases, it's harder to find the best values
4. Use colsample_bynode in range 0.5-0.8
5. Use subsample in range 0.7-0.8
6. Set L2 reg to about 30 (depends on magnitude of labels, mine were about 1-10 so maybe 3 for 0-1 labels)
7. L1 reg is not very important, but helps find better L2 values

## Conclusions

1. When you're doing in-time validation, train and val errors will be very closely coupled and it's hard to overfit or do poorly (e.g. even 400 depth and 400 trees is fine).
2. For in-time validation, none of the hyperparameters matter too much except lr and maybe subsample.
3. Increasing data from 70,000 to 700,000 improves performance 0.42 to 0.24.
4. Unclear if performance increases more when we use 7,000,000 samples.
5. OOT validation leads to overfitting. Overfitting causes the *range of viable hyperparameters to narrow*.
6. A quick solution to overfitting is using shallower trees and more rounds helps. This is *not as good as the next solution*.
7. The *real solution* to solve overfitting is to use *deeper trees* and *more rounds* (overfit more) then apply regularization. I found that you need approximately 2.5x more trees and make them 2.5x deeper. 
8. Ranked list of most important features to regularize: lr, L2 reg, colsample_bynode, subsample, L1 reg.
9. L2 does the job by itself, L1 is not necesssary but may help find optimal L2 values easier.
10. Some guidelines on hyperparameters to use (obviously will depend on how much overfitting is happening): trees=80, depth=25, lr=0.1, L2 = 3x max y-value, colsample_bynode=0.5-0.8, subsample=0.7-0.8, L1=L2.
11. The DART model works worse than XGBoost out of the box (even though dropout is set to 0), this implies there's some bug or some difference that in the DART implementation. In any case, DART is probably not useful.
12. Adding trees generally doesn't overfit, it just takes longer to compute.
