import XCTest
@testable import SwiftNaiveBayes

class SwiftNaiveBayesTests: XCTestCase {
    func testExample() {
        // This is an example of a functional test case.
        // Use XCTAssert and related functions to verify your tests produce the correct results.
        XCTAssertEqual(SwiftNaiveBayes().text, "Hello, World!")
    }


    static var allTests : [(String, (SwiftNaiveBayesTests) -> () throws -> Void)] {
        return [
            ("testExample", testExample),
        ]
    }
}
