For SGD, lr_decay does not have significant effect. NO lr_decay is better.
For SGD, increase lr would accelerate decrease, with the same final loss; decrease lr makes the final loss worse. -> lr=0.01

For SGD lr=0.01, SGD_HD does not help SGD.
For SGD lr=0.0001, SGD_HD helps SGD to have the rate as 0.01.
SGD_HDN not helps, either.

diagonal scaling helps? NO.
momentum helps? NO.

Adam_HD helps?
Adam also stucks, only AdamW works.

turn-off mixed-precision? NO. same samples not help.

weight_decay is very important (only for Adam)! Adam (wd=0.1) stucks, but wd=1e-4 decreases very well.

num decayed parameter tensors: 26, with 10,740,096 parameters
num non-decayed parameter tensors: 13, with 4,992 parameters