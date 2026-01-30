import { Certificate } from '../types'

interface CertificateViewProps {
  certificate: Certificate
  onRestart: () => void
}

function CertificateView({ certificate, onRestart }: CertificateViewProps) {
  const handlePrint = () => {
    window.print()
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-900 via-purple-900 to-pink-800 flex flex-col items-center justify-center p-6">
      <div
        id="certificate"
        className="max-w-3xl w-full bg-gradient-to-br from-amber-50 to-orange-100 rounded-3xl p-12 shadow-2xl border-8 border-amber-400/50 relative overflow-hidden"
      >
        <div className="absolute top-0 left-0 w-32 h-32 bg-amber-400/20 rounded-full -translate-x-1/2 -translate-y-1/2" />
        <div className="absolute bottom-0 right-0 w-48 h-48 bg-amber-400/20 rounded-full translate-x-1/4 translate-y-1/4" />
        <div className="absolute top-1/2 left-0 w-2 h-32 bg-amber-400/30 -translate-y-1/2" />
        <div className="absolute top-1/2 right-0 w-2 h-32 bg-amber-400/30 -translate-y-1/2" />

        <div className="relative text-center">
          <div className="mb-8">
            <span className="text-amber-700 text-sm font-semibold tracking-[0.3em] uppercase">
              Self Training Machine
            </span>
          </div>

          <h1 className="text-5xl font-serif text-amber-900 mb-6">
            Certificate of Completion
          </h1>

          <div className="text-gray-600 mb-8">This certifies that</div>

          <div className="text-4xl font-bold text-amber-800 mb-8 font-serif">
            {certificate.user_name}
          </div>

          <div className="text-gray-600 mb-4">has successfully completed the training</div>

          <div className="text-2xl font-semibold text-amber-700 mb-8 px-8">
            "{certificate.training_title}"
          </div>

          <div className="flex justify-center gap-16 mb-10">
            <div className="text-center">
              <div className="text-4xl font-bold text-green-600">{certificate.score}</div>
              <div className="text-gray-500 text-sm">Score</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-amber-600">{certificate.total}</div>
              <div className="text-gray-500 text-sm">Total</div>
            </div>
            <div className="text-center">
              <div className="text-4xl font-bold text-purple-600">
                {certificate.percentage.toFixed(0)}%
              </div>
              <div className="text-gray-500 text-sm">Grade</div>
            </div>
          </div>

          <div className="border-t-2 border-amber-300 pt-6 mt-6">
            <div className="flex justify-between items-end">
              <div className="text-left">
                <div className="text-gray-500 text-sm">Date</div>
                <div className="text-amber-800 font-semibold">{certificate.date}</div>
              </div>
              <div className="text-center">
                <div className="w-32 h-32 mx-auto mb-2 relative">
                  <div className="absolute inset-0 border-4 border-amber-400 rounded-full" />
                  <div className="absolute inset-2 border-2 border-amber-300 rounded-full flex items-center justify-center">
                    <span className="text-amber-600 text-xs font-bold">VERIFIED</span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="text-gray-500 text-sm">Certificate ID</div>
                <div className="text-amber-800 font-mono text-xs">
                  {certificate.id.slice(0, 8).toUpperCase()}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="flex gap-4 mt-8 print:hidden">
        <button
          onClick={handlePrint}
          className="px-8 py-3 bg-amber-500 text-white font-bold rounded-xl hover:bg-amber-600 transition-all shadow-lg"
        >
          Print Certificate
        </button>
        <button
          onClick={onRestart}
          className="px-8 py-3 bg-purple-500 text-white font-bold rounded-xl hover:bg-purple-600 transition-all shadow-lg"
        >
          Start New Training
        </button>
      </div>
    </div>
  )
}

export default CertificateView
