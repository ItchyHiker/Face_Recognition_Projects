## 问题
1. 模型不收敛，loss不变化：由SGD换成Adam优化器解决问题
2. LFW: Marilyn_Monroe/Marilyn_Monroe_0001.jpg 没有切出来
3. LFW accuracy 到 82% 左右就不提升了
	+ 查看 eval 是否有问题
	+ 貌似是LFW图没有切好或者是数据读取不对
	问题就是图没有切好



## Todo
- [x] Test
- [x] Save Model
- [ ] Save best model
- [ ] Transforms
- [ ] Logs
- [ ] Loss functions
	- [x] Softmax
	- [ ] NormFace
	- [ ] SphereFace
- [ ] Flip while evaluating
- [ ] L2 Distance bug
- [x] MobileFaceNet
- [x] Load pretrained weights
- [ ] Plot metric space
- [x] Compare LFW cropped result with other method



## Reference
1. Losses: https://github.com/DequanZhu/FaceNet-and-FaceLoss-collections-tensorflow2.0
2. Plot metric space: https://github.com/cvqluu/Angular-Penalty-Softmax-Losses-Pytorch
