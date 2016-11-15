import XCTest
@testable import SwiftNaiveBayes

class NaiveBayesTests: XCTestCase {
    
    let pos = ["computer": 3, "programming": 2, "python": 1, "swift": 2]
    let neg = ["game": 2, "computer": 2, "video": 1, "programming": 1]
    let posTest = ["computer": 2, "ruby": 1, "swift": 1, "programming": 1]
    
    override func setUp() {
        super.setUp()
    }
    
    override func tearDown() {
        super.tearDown()
    }

    func testFit() {
        let nb = NaiveBayes()
        let result = nb.fit(["positive": pos, "negative": neg]).predict(posTest)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFit2() {
        let nb = NaiveBayes()
        let _ = nb.addInstance(pos, label: "positive")
        let _ = nb.addInstance(neg, label: "negative")
        let result = nb.fit().predict(posTest)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFit3() {
        let nb = NaiveBayes()
        let _ = nb.addInstances(["positive": pos, "negative": neg]).fit()
        let result = nb.predict(posTest)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFit4() {
        let nb = NaiveBayes(["positive": pos, "negative": neg]).fit()
        let result = nb.predict(posTest)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFileIO1() {
        do {
            let nbA = NaiveBayes()
            let resultA = nbA.fit(["positive": pos, "negative": neg]).predict(posTest)
            try nbA.save("/tmp/nbA.session")
            
            let nbB = NaiveBayes()
            try nbB.restore("/tmp/nbA.session")
            let resultB = nbB.predict(posTest)

            for (label, score) in resultA {
                XCTAssertEqual(score, resultB[label])
            }
        } catch {
            XCTFail()
        }
    }
    
    func testFileIO2() {
        do {
            let nbA = NaiveBayes()
            let resultA = nbA.fit(["positive": pos, "negative": neg]).predict(posTest)
            try nbA.save("/tmp/nbA.session")
            
            let nbB = try NaiveBayes("/tmp/nbA.session")
            let resultB = nbB.predict(posTest)
            
            for (label, score) in resultA {
                XCTAssertEqual(score, resultB[label])
            }
        } catch {
            XCTFail()
        }
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
