![SwiftNaiveBayes](/logo/logo.png)

[![](http://img.shields.io/badge/Swift-3.0-blue.svg)]()![macOS](https://img.shields.io/badge/os-macOS-green.svg?style=flat)
![Linux](https://img.shields.io/badge/os-linux-green.svg?style=flat)[![Carthage compatible](https://img.shields.io/badge/Carthage-compatible-4BC51D.svg?style=flat)](https://github.com/Carthage/Carthage) [![Swift Package Manager compatible](https://img.shields.io/badge/Swift%20Package%20Manager-compatible-brightgreen.svg)](https://github.com/apple/swift-package-manager)[![CocoaPods compatible](https://img.shields.io/badge/CocoaPods-compatible-4BC51D.svg)](https://github.com/CocoaPods/CocoaPods)[![MIT License](http://img.shields.io/badge/license-MIT-blue.svg?style=flat)](LICENSE)

Naive Bayes classifier implemented in Swift

## Description

Implementation of Naive Bayes Classifier algorithm in Swift3.x, supporting **Gaussian naive Bayes**, **Multinomial naive Bayes** and **Bernoulli naive Bayes**

## Usage

SwiftNaiveBayes is same as [Scikit learn](http://scikit-learn.org/stable/modules/naive_bayes.html)'s interface.

```swift
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
let logProbs = nb.predict(posTest)
print(logProbs) //=> ["positive": -8.9186205290602363, "negative": -10.227308671603783]
// Use method chaining
nb.fit(["positive": pos, "negative": neg]).predict(posTest)

// Save session
try! nb.save("nb.session")
// Restore session
let nb2 = NaiveBayes("nb.session")
```

## Install

### Swift Package Manager

`Package.swift`:

```swift
import PackageDescription

let package = Package(
    name: "MyApp",
    targets: [],
    dependencies: [
        .Package(url: "https://github.com/akimach/SwiftNaiveBayes.git", majorVersion: 1),
    ]
)
```

### CocoaPods

`Podfile`:

```
platform :ios, '8.0'
use_frameworks!

target 'MyApp' do
    pod 'SwiftNaiveBayes'
end
```

### Carthage

`Cartfile`:

```
github "akimach/SwiftNaiveBayes"
```

## Licence

[MIT](https://github.com/akimach/SwiftNaiveBayes/blob/master/LICENSE)

## Author

[akimach](https://github.com/akimach)
