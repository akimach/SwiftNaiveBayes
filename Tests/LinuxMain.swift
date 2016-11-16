import XCTest
@testable import NaiveBayesTests
@testable import GaussianNaiveBayesTests
@testable import BernoulliNaiveBayesTests

XCTMain([
     testCase(NaiveBayesTests.allTests),
     testCase(GaussianNaiveBayesTests.allTests),
     testCase(BernoulliNaiveBayesTests.allTests),
])
