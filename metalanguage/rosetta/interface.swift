import SwiftUI

struct ProtoAGIView: View {
    var integrationPhi: Double
    var body: some View {
        Text("System Synergy: \(integrationPhi)")
            .foregroundColor(integrationPhi > 0.618 ? .green : .red)
    }
}
