import SwiftUI

struct ControllerView: View {
    @EnvironmentObject private var controller: MotionController
    @FocusState private var focused: Bool

    private let ink = Color(red: 0.09, green: 0.07, blue: 0.12)
    private let cream = Color(red: 1, green: 0.94, blue: 0.79)
    private let coral = Color(red: 1, green: 0.36, blue: 0.30)
    private let yellow = Color(red: 1, green: 0.83, blue: 0.28)
    private let mint = Color(red: 0.40, green: 0.84, blue: 0.65)

    var body: some View {
        ZStack {
            ink.ignoresSafeArea()
            VStack(spacing: 18) {
                title
                connectionPanel
                motionPanel
                snapButton
                Text("Tilt to aim. Push the phone forward quickly to snap.")
                    .font(.system(size: 13, weight: .bold, design: .monospaced))
                    .foregroundStyle(cream.opacity(0.72))
                    .multilineTextAlignment(.center)
                    .padding(.horizontal, 24)
            }
            .padding(20)
        }
        .preferredColorScheme(.dark)
    }

    private var title: some View {
        VStack(spacing: 5) {
            Text("KITCHEN PATROL UNIT 84")
                .font(.system(size: 12, weight: .black, design: .monospaced))
                .tracking(2)
                .foregroundStyle(yellow)
            Text("FLY CATCHER")
                .font(.system(size: 40, weight: .black, design: .rounded))
                .foregroundStyle(cream)
                .shadow(color: coral, radius: 0, x: 4, y: 4)
        }
    }

    private var connectionPanel: some View {
        VStack(spacing: 12) {
            HStack(spacing: 10) {
                Circle()
                    .fill(controller.active ? mint : coral)
                    .frame(width: 12, height: 12)
                    .shadow(color: controller.active ? mint : coral, radius: 8)
                Text(controller.status.uppercased())
                    .font(.system(size: 12, weight: .black, design: .monospaced))
                    .foregroundStyle(cream)
                Spacer()
            }
            HStack(spacing: 10) {
                TextField("Mac IPv4 address", text: $controller.host)
                    .textInputAutocapitalization(.never)
                    .keyboardType(.numbersAndPunctuation)
                    .autocorrectionDisabled()
                    .focused($focused)
                    .disabled(controller.active)
                    .padding(13)
                    .foregroundStyle(ink)
                    .background(cream)
                TextField("Port", text: $controller.port)
                    .keyboardType(.numberPad)
                    .focused($focused)
                    .disabled(controller.active)
                    .frame(width: 72)
                    .padding(13)
                    .foregroundStyle(ink)
                    .background(cream)
            }
            Button(controller.active ? "STOP LINK" : "START UDP LINK") {
                focused = false
                controller.active ? controller.stop() : controller.start()
            }
            .buttonStyle(PixelButtonStyle(background: controller.active ? coral : mint, foreground: ink))
        }
        .padding(16)
        .background(Color(red: 0.18, green: 0.14, blue: 0.21))
        .overlay(Rectangle().stroke(Color(red: 0.32, green: 0.23, blue: 0.34), lineWidth: 3))
    }

    private var motionPanel: some View {
        HStack(spacing: 8) {
            meter("X", controller.acceleration.x)
            meter("Y", controller.acceleration.y)
            meter("Z", controller.acceleration.z)
        }
    }

    private func meter(_ name: String, _ value: Double) -> some View {
        VStack(spacing: 7) {
            Text(name)
                .foregroundStyle(coral)
            Text(value.formatted(.number.precision(.fractionLength(2))))
                .foregroundStyle(yellow)
        }
        .font(.system(size: 15, weight: .black, design: .monospaced))
        .frame(maxWidth: .infinity)
        .padding(.vertical, 13)
        .background(Color.black.opacity(0.28))
        .overlay(Rectangle().stroke(Color(red: 0.32, green: 0.23, blue: 0.34), lineWidth: 2))
    }

    private var snapButton: some View {
        Button {
            controller.sendSnap()
        } label: {
            ZStack {
                RoundedRectangle(cornerRadius: 24)
                    .fill(controller.active ? coral : Color.gray)
                RoundedRectangle(cornerRadius: 24)
                    .stroke(ink, lineWidth: 7)
                Text("SNAP")
                    .font(.system(size: 44, weight: .black, design: .rounded))
                    .foregroundStyle(cream)
                    .shadow(color: ink, radius: 0, x: 4, y: 4)
            }
            .frame(height: 150)
        }
        .buttonStyle(.plain)
        .disabled(!controller.active)
    }
}

struct PixelButtonStyle: ButtonStyle {
    let background: Color
    let foreground: Color

    func makeBody(configuration: Configuration) -> some View {
        configuration.label
            .font(.system(size: 15, weight: .black, design: .monospaced))
            .frame(maxWidth: .infinity)
            .padding(.vertical, 14)
            .foregroundStyle(foreground)
            .background(background)
            .offset(x: configuration.isPressed ? 3 : 0, y: configuration.isPressed ? 3 : 0)
    }
}
