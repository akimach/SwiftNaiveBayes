import Foundation

public enum NaiveBayesError: Error {
    case FileIO(msg: String)
    case UnknownValue
}

public class NaiveBayes {
    
    // MARK: - Private properties
    
    /// All attributes
    private var attributes: [String:Int] = [:]
    /// attributes in Labels
    private var labelAttrs: [String:[String:Int]] = [:]
    /// P(attr|label)
    private var attrProbs: [String:[String:Double]] = [:]
    /// To avoid zero-frequency problem.
    private var smoother: [String:Double] = [:]
    
    // MARK: - Pubilc methods
    
    public init() {}
    
    public init(_ dataset: [String: [String:Int]]) {
        let _ = self.addInstances(dataset)
    }
    
    public init(_ path: String) throws {
        do {
            try self.restore(path)
        } catch (let e) {
            throw e
        }
    }
    
    public func addInstance(_ attr: [String:Int], label: String) -> Self {
        self.labelAttrs[label] = self.labelAttrs[label] ?? [:]
        for (k, v) in attr {
            // coutup attributes
            self.attributes[k] = (self.attributes[k] ?? 0) + v
            // countup attributes in label
            self.labelAttrs[label]?[k] = (self.labelAttrs[label]?[k] ?? 0) + v
        }
        return self
    }
    
    public func addInstances(_ dataset: [String: [String:Int]]) -> Self {
        // Setup attributes
        for (label, attr) in dataset {
            let _  = self.addInstance(attr, label: label)
        }
        return self
    }
    
    public func fit() -> Self {
        // Calculate P(attr|label) for each label
        for (label, attrs) in self.labelAttrs {
            self.attrProbs[label] = [:]
            self.smoother[label] = -log(self.denom(label))
            for (token, value) in attrs {
                self.attrProbs[label]?[token] = log(Double(value)+1) - log(self.denom(label))
            }
        }
        return self
    }
    
    public func fit(_ dataset: [String: [String:Int]]) -> Self {
        return self.addInstances(dataset).fit()
    }
    
    public func predict(_ attrs: [String:Int]) -> [String:Double] {
        var result: [String:Double] = [:]
        for label in self.labelAttrs.keys {
            let prob = self.score(attrs, label: label)
            result[label] = prob
        }
        return result
    }
    
    public func save(_ path: String) throws {
        let targets = [self.attributes, self.labelAttrs, self.attrProbs, self.smoother] as [Any]
        var rawDataList: [Data] = []
        for target in targets {
            let data: Data = NSKeyedArchiver.archivedData(withRootObject: target)
            rawDataList.append(data)
        }
        do {
            let data = NSKeyedArchiver.archivedData(withRootObject: rawDataList)
            try data.write(to: URL(fileURLWithPath: path))
        } catch (let e) {
            throw NaiveBayesError.FileIO(msg: e.localizedDescription)
        }
    }
    
    public func restore(_ path: String) throws {
        do {
            let url = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: url)
            let unarchivedData = NSKeyedUnarchiver.unarchiveObject(with: data) as! [Any]
            self.attributes = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[0] as! Data) as! [String:Int]
            self.labelAttrs = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[1] as! Data) as! [String:[String:Int]]
            self.attrProbs = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[2] as! Data) as! [String:[String:Double]]
            self.smoother = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[3] as! Data) as! [String:Double]
        } catch let e {
            throw NaiveBayesError.FileIO(msg: e.localizedDescription)
        }
    }
    
    // MARK: - Private methods
    
    /*
     * Denominator of P(attr|label)
     */
    private func denom(_ label: String) -> Double {
        guard let lAttrs = self.labelAttrs[label] else {
            return 1.0
        }
        let values = Array(lAttrs.values)
        let attrSize = Double(self.attributes.count)
        let labelAttrSize = Double(values.reduce(0, +))
        return labelAttrSize + attrSize
    }
    
    /*
     * Prior Probability: P(label)
     */
    private func priorProb(_ label: String) -> Double {
        return 1.0 / Double(self.labelAttrs.count)
    }
    
    /*
     * Log of P(label|attrs)
     */
    private func score(_ newAttrs: [String:Int], label: String) -> Double {
        var score = log(self.priorProb(label)) // Prior prob = P(label)
        for attr in newAttrs.keys {
            let sm = self.smoother[label] ?? 0.0
            score += (self.attrProbs[label]?[attr] ?? sm) * Double(newAttrs[attr] ?? 1)
        }
        return score
    }
}

public typealias MultinomialNaiveBayes = NaiveBayes

class GaussianNaiveBayes {
    
    // MARK: - Private properties
    
    /// attributes in Labels
    private var labelAttrs: [String:[Double]] = [:]
    /// mean of label
    private var labelMean: [String:Double] = [:]
    /// var of label
    private var labelVar: [String:Double] = [:]
    
    // MARK: - Public methods
    
    public init() {}
    
    public init(_ dataset: [String:[Double]]) {
        let _ = self.addInstances(dataset)
    }
    
    public init(_ path: String) throws {
        do {
            try self.restore(path)
        } catch (let e) {
            throw e
        }
    }
    
    public func addInstance(_ attrs: [Double], label: String) -> Self {
        self.labelAttrs[label] = (self.labelAttrs[label] ?? []) + attrs
        return self
    }
    
    public func addInstances(_ dataset: [String:[Double]]) -> Self {
        for (label, attr) in dataset {
            let _  = self.addInstance(attr, label: label)
        }
        return self
    }
    
    public func fit() -> Self {
        for label in self.labelAttrs.keys {
            let values = self.labelAttrs[label] ?? []
            self.labelMean[label] = self.mean(values)
            self.labelVar[label] = self.variance(values)
        }
        return self
    }
    
    public func fit(_ dataset: [String:[Double]]) -> Self {
        return self.addInstances(dataset).fit()
    }
    
    public func predict(_ newAttrs: [Double]) -> [String:Double] {
        var result: [String:Double] = [:]
        for label in self.labelAttrs.keys {
            result[label] = self.score(newAttrs, label: label)
        }
        return result
    }
    
    public func save(_ path: String) throws {
        let targets = [self.labelAttrs, self.labelMean, self.labelVar] as [Any]
        var rawDataList: [Data] = []
        for target in targets {
            let data: Data = NSKeyedArchiver.archivedData(withRootObject: target)
            rawDataList.append(data)
        }
        do {
            let data = NSKeyedArchiver.archivedData(withRootObject: rawDataList)
            try data.write(to: URL(fileURLWithPath: path))
        } catch (let e) {
            throw NaiveBayesError.FileIO(msg: e.localizedDescription)
        }
    }
    
    public func restore(_ path: String) throws {
        do {
            let url = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: url)
            let unarchivedData = NSKeyedUnarchiver.unarchiveObject(with: data) as! [Any]
            self.labelAttrs = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[0] as! Data) as! [String:[Double]]
            self.labelMean = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[1] as! Data) as! [String:Double]
            self.labelVar = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[2] as! Data) as! [String:Double]
        } catch let e {
            throw NaiveBayesError.FileIO(msg: e.localizedDescription)
        }
    }
    
    // MARK: - Private methods
    
    /*
     * Prior probability, P(label)
     */
    private func priorProb(_ label: String) -> Double {
        let cnt = Double(self.labelAttrs.count)
        return 1.0 / cnt
    }
    
    /*
     * Log of P(label|attrs)
     */
    private func score(_ attrs: [Double], label: String) -> Double {
        guard let mean = self.labelMean[label] else {return -Double.infinity}
        guard let sigma = self.labelVar[label] else {return -Double.infinity}
        var score = log(self.priorProb(label)) // Prior prob
        for attr in attrs {
            let val = pow(attr - mean, 2.0) / sigma
            score -= 0.5 * (log(sigma) + log(2.0*M_PI) + val)
        }
        return score
    }
    
    private func mean(_ values: [Double]) -> Double {
        if values.count == 0 { return 0.0 }
        let n = Double(values.count)
        let sum = values.reduce(0.0, +)
        return sum / n
    }
    
    private func variance(_ values: [Double]) -> Double {
        if values.count == 0 { return 0.0 }
        let n = Double(values.count)
        let sum = values.reduce(0.0, +)
        let mean = sum / n
        let var_s = values.reduce(0.0) { (result, val) in
            return result + pow(val-mean, 2.0)
        }
        return var_s / n
    }
}

public class BernoulliNaiveBayes {
    
    // MARK: - Private properties
    
    /// attributes in Labels
    private var labelAttrs: [String:[Int]] = [:]
    /// probavility of positive
    private var posProbs: [String:Double] = [:]
    
    // MARK: - Public methods
    
    init() { }
    
    init(_ dataset: [String:[Int]]) throws {
        do {
            let _ = try self.addInstances(dataset)
        } catch {
            throw NaiveBayesError.UnknownValue
        }
    }
    
    init(_ path: String) throws {
        do {
            try self.restore(path)
        } catch (let e) {
            throw e
        }
    }
    
    public func addInstance(_ attrs: [Int], label: String) throws -> Self {
        let f = attrs.filter {$0 != 1 && $0 != 0}
        if f.count != 0 {
            throw NaiveBayesError.UnknownValue
        }
        
        self.labelAttrs[label] = (self.labelAttrs[label] ?? []) + attrs
        return self
    }
    
    public func addInstances(_ dataset: [String:[Int]]) throws -> Self {
        for (label, attrs) in dataset {
            guard let _ = try? self.addInstance(attrs, label: label) else {
                throw NaiveBayesError.UnknownValue
            }
        }
        return self
    }
    
    public func fit() -> Self {
        for (label, attrs) in self.labelAttrs {
            let pos = Double(attrs.reduce(0, +))
            let cnt = Double(attrs.count)
            self.posProbs[label] = pos / cnt
        }
        return self
    }
    
    public func fit(_ dataset: [String:[Int]]) throws -> Self {
        guard let _ =  try? self.addInstances(dataset) else {
            throw NaiveBayesError.UnknownValue
        }
        return self.fit()
    }
    
    public func predict(_ newAttrs: [Int]) -> [String:Double] {
        var result: [String:Double] = [:]
        for label in self.labelAttrs.keys {
            result[label] = self.score(newAttrs, label: label)
        }
        return result
    }
    
    public func save(_ path: String) throws {
        let targets = [self.labelAttrs, self.posProbs] as [Any]
        var rawDataList: [Data] = []
        for target in targets {
            let data: Data = NSKeyedArchiver.archivedData(withRootObject: target)
            rawDataList.append(data)
        }
        do {
            let data = NSKeyedArchiver.archivedData(withRootObject: rawDataList)
            try data.write(to: URL(fileURLWithPath: path))
        } catch (let e) {
            throw NaiveBayesError.FileIO(msg: e.localizedDescription)
        }
    }
    
    public func restore(_ path: String) throws {
        do {
            let url = URL(fileURLWithPath: path)
            let data = try Data(contentsOf: url)
            let unarchivedData = NSKeyedUnarchiver.unarchiveObject(with: data) as! [Any]
            self.labelAttrs = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[0] as! Data) as! [String:[Int]]
            self.posProbs = NSKeyedUnarchiver.unarchiveObject(with: unarchivedData[1] as! Data) as! [String:Double]
        } catch let e {
            throw NaiveBayesError.FileIO(msg: e.localizedDescription)
        }
    }
    
    // MARK: - Private methods
    
    /*
     * Kronecker delta
     */
    private func delta(_ i: Double, _ j: Double) -> Double {
        return (i == j) ? 1.0 : 0.0
    }
    
    /*
     * P(label)
     */
    private func priorProb(_ label: String) -> Double {
        return 1.0 / Double(self.labelAttrs.count)
    }
    
    private func score(_ attrs: [Int], label: String) -> Double {
        var score = log(self.priorProb(label)) // Prior prob
        for attr in attrs {
            guard let posProb = posProbs[label] else { return -Double.infinity }
            let x = Double(attr)
            score += log(self.delta(1, x) * posProb + self.delta(0, x) * (1 - posProb))
        }
        return score
    }
}
