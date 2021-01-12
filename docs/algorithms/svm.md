# SVM algorithm

## Defination
- Supervised learning methods used for [classification](https://scikit-learn.org/stable/modules/svm.html#svm-classification), [regression](https://scikit-learn.org/stable/modules/svm.html#svm-regression)and [outliers detection](https://scikit-learn.org/stable/modules/svm.html#svm-outlier-detection)
- Different [Kernel functions](https://scikit-learn.org/stable/modules/svm.html#svm-kernels) can be specified for the decision function
- Some type of SVM: SVC and NuSVC (one-versus-one), LinearSVC (sklearn)
- SVM còn được coi là maximum margin classifier.
- Sử dụng Karus Kuln Tucker

## Some questions
- Why maximize margin?
Cần hiểu về khái niệm công bằng (các điểm được phân chia đồng đều - bằng nhau đến đường phân chia), và thịnh vượng (khoảng cách đến hyperplane cần max, khi đó cả 2 class đều vui)
- Các chứng minh dùng Largrage khá là khó hiểu, có cách nào intuative hơn không?
Phải hiểu, có link hay đây :v (below)
- Does Slack and C are hyperparameter?

## Support topics

### Linear Separator
- We have perceptron, why on earth we bother on this?
- In LS, can we pick 50 instead of 1 => its not a problem, choose any number (but equal in both side)

### Large Margin Classifier
- Basically, this is optimization problem
- Updating math function here .…
- Nếu gặp trường hợp point quá lỗi => sử dụng slack


