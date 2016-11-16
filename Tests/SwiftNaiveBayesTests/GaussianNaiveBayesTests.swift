//
//  GaussianNaiveBayesTests.swift
//  SwiftNaiveBayes
//
//  Created by akimach on 2016/11/14.
//
//

import XCTest
@testable import SwiftNaiveBayes

class GaussianNaiveBayesTests: XCTestCase {

    /// Z(10, 5) Mean = 10, Variance = 5
    let posZ = [14.85745575,   1.66435772,   5.4901721 ,  10.27362972,
                2.04665928,   6.34189749,   2.36884985,   5.83045205,
                7.82074239,  10.26142184,  12.20555032,   9.09912813,
                10.96364953,  11.2820095 ,   0.72612828,   8.92463179,
                13.30905535,   9.06589405,   5.76409129,   8.94796974]
    
    /// Z(0, 1) Mean = 0, Variance = 1
    let negZ = [-1.37882159,  0.88735308, -1.70868543,  1.41404701,  0.60389257,
                -0.64336808, -0.54142288, -2.84982192,  1.85876875,  0.05157414,
                0.33596444,  0.72804517,  0.65131547, -0.09489481,  0.85194423,
                0.19820711,  1.44428983,  0.49041096, -1.31409282, -0.24759202]
    
    /// Z(10, 5) Mean = 10, Variance = 5
    let testPosZ = [4.34736675,  12.94046014,   7.56662414,  13.55149247,
                    13.99427358,  10.04878958,  16.87970498,  10.27722349,
                    6.76021957,   4.28015561]
    
    override func setUp() {
        super.setUp()
    }
    
    override func tearDown() {
        super.tearDown()
    }

    func testFit1() {
        let gnb = GaussianNaiveBayes()
        let result = gnb.fit(["positive": posZ, "negative": negZ]).predict(testPosZ)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFit2() {
        let gnb = GaussianNaiveBayes(["positive": posZ, "negative": negZ]).fit()
        let result = gnb.predict(testPosZ)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFit3() {
        let gnb = GaussianNaiveBayes()
        _ = gnb.addInstance(posZ, label: "positive")
        _ = gnb.addInstance(negZ, label: "negative")
        let result = gnb.fit().predict(testPosZ)
        XCTAssertEqual(self.bestLabel(result), "positive")
    }
    
    func testFileIO1() {
        do {
            let gnbA = GaussianNaiveBayes()
            let resultA = gnbA.fit(["positive": posZ, "negative": negZ]).predict(testPosZ)
            try gnbA.save("/tmp/nbA.session")
            let gnbB = GaussianNaiveBayes()
            try gnbB.restore("/tmp/nbA.session")
            let resultB = gnbB.predict(testPosZ)
            for (label, score) in resultA {
                XCTAssertEqual(score, resultB[label])
            }
        } catch {
            XCTFail()
        }
    }
    
    func testFileIO2() {
        do {
            let gnbA = GaussianNaiveBayes()
            let resultA = gnbA.fit(["positive": posZ, "negative": negZ]).predict(testPosZ)
            try gnbA.save("/tmp/bnbA.session")
            let gnbB = try GaussianNaiveBayes("/tmp/bnbA.session")
            let resultB = gnbB.predict(testPosZ)
            for (label, score) in resultA {
                XCTAssertEqual(score, resultB[label])
            }
        } catch {
            XCTFail()
        }
    }
    
    static var allTests : [(String, (GaussianNaiveBayesTests) -> () throws -> Void)] {
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
