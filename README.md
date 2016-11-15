# SwiftNaiveBayes

Naive Bayes classifier implemented in Swift

## Description

Implementation of Naive Bayes Classifier algorithm in Swift3.x, supporting **Gaussian naive Bayes**, **Multinomial naive Bayes** and **Bernoulli naive Bayes**

## Usage

```
import SwiftNaiveBayes

let nb = NaiveBayes()
// Positive tokens and the frequencies ["token A": Frequency of token A, ...]
let pos = ["computer": 3, "programming": 2, "python": 1, "swift": 2]
// Negative tokens ["token X": Frequency of token X, ...]
let neg = ["game": 2, "computer": 2, "video": 1, "programming": 1]
// Positive tokens for testing
let posTest = ["computer": 2, "ruby": 1, "swift": 1, "programming": 1]
// Train model
// ["Label A": ["token A": Frequency of token A, ...]]
nb.fit(["positive": pos, "negative": neg])
// Predicts log probabilities for each label
nb.predict(posTest)

// Save session
try! nb.save("nb.session")
// Restore session
let nb2 = NaiveBayes("nb.session")
```

## Install

### Swift Package Manager

Package.swift

```swift 
import PackageDescription

let package = Package(
    name: "YourProjectName",
    targets: [],
    dependencies: [
        .Package(url: "https://github.com/akimach/SwiftNaiveBayes.git", majorVersion: 1),
    ]
)
```

Run `swift build`.

### CocoaPods

### Carthage

## Licence

[MIT](https://github.com/akimach/SwiftNaiveBayes/blob/master/LICENSE)

## Author

[akimach](https://github.com/akimach)
