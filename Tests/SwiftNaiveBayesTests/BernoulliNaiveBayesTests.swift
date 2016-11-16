import XCTest
@testable import SwiftNaiveBayes

class BernoulliNaiveBayesTests: XCTestCase {
    /// N(n, 0.2)
    let pos = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1,
               0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
               1, 0, 0, 0]
    /// N(n, 0.8)
    let neg = [1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1,
               0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 1,
               0, 0, 1, 1]
    /// N(n, 0.2)
    let posTest = [1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0]
    
    override func setUp() {
        super.setUp()
    }
    
    override func tearDown() {
        super.tearDown()
    }
    
    func testFit1() {
        do {
            let bnb = BernoulliNaiveBayes()
            let result = try bnb.fit(["positive": pos, "negative": neg]).predict(posTest)
            XCTAssertEqual(self.bestLabel(result), "positive")
        } catch {
            XCTFail()
        }
    }
    
    func testFit2() {
        do {
            let bnb = try BernoulliNaiveBayes(["positive": pos, "negative": neg]).fit()
            let result = bnb.predict(posTest)
            XCTAssertEqual(self.bestLabel(result), "positive")
        } catch {
            XCTFail()
        }
    }
    
    func testFit3() {
        do {
            let bnb = BernoulliNaiveBayes()
            _ = try bnb.addInstance(pos, label: "positive")
            _ = try bnb.addInstance(neg, label: "negative")
            let result = bnb.predict(posTest)
            XCTAssertEqual(self.bestLabel(result), "positive")
        } catch {
            XCTFail()
        }
    }
    
    func testFileIO1() {
        do {
            let bnbA = BernoulliNaiveBayes()
            let resultA = try bnbA.fit(["positive": pos, "negative": neg]).predict(posTest)
            try bnbA.save("/tmp/bnbA.session")
            let bnbB = BernoulliNaiveBayes()
            try bnbB.restore("/tmp/bnbA.session")
            let resultB = bnbB.predict(posTest)
            for (label, score) in resultA {
                XCTAssertEqual(score, resultB[label])
            }
        } catch {
            XCTFail()
        }
    }
    
    func testFileIO2() {
        do {
            let bnbA = BernoulliNaiveBayes()
            let resultA = try bnbA.fit(["positive": pos, "negative": neg]).predict(posTest)
            try bnbA.save("/tmp/bnbA.session")
            let bnbB = try BernoulliNaiveBayes("/tmp/bnbA.session")
            let resultB = bnbB.predict(posTest)
            for (label, score) in resultA {
                XCTAssertEqual(score, resultB[label])
            }
        } catch {
            XCTFail()
        }
    }
    
    static var allTests : [(String, (BernoulliNaiveBayesTests) -> () throws -> Void)] {
        return [
            ("testFit1", testFit1),
            ("testFit2", testFit2),
            ("testFit3", testFit3),
            ("testFileIO1", testFileIO1),
            ("testFileIO2", testFileIO2),
        ]
    }
    
    private func bestLabel(_ result: [String:Double]) -> String {
        var bestLabel: String = (result.first?.key)!
        var bestScore: Double = (result.first?.value)!
        for (label, score) in result {
            if score > bestScore {
                (bestLabel, bestScore) = (label, score)
            }
        }
        return bestLabel
    }
}
