Pod::Spec.new do |s|
  s.name        = "SwiftNaiveBayes"
  s.version     = "1.0.1"
  s.summary     = "Naive Bayes classifier implemented in Swift"
  s.homepage    = "https://github.com/akimach/SwiftNaiveBayes"
  s.license     = { :type => "MIT" }
  s.authors     = { "akimach" => "kimura.akimasa@gmail.com"}

  s.requires_arc = true
  s.osx.deployment_target = "10.9"
  s.ios.deployment_target = "8.0"
  s.watchos.deployment_target = "2.0"
  s.tvos.deployment_target = "9.0"
  s.source   = { :git => "https://github.com/akimach/SwiftNaiveBayes.git", :tag => "v#{s.version}" }
  s.source_files = "Sources/*.swift"
  s.pod_target_xcconfig =  {
        'SWIFT_VERSION' => '3.0',
  }
end