import { useState, useCallback } from "react";
import { Upload, Loader2, CheckCircle2, AlertTriangle, Leaf, RotateCcw } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FloatingShapes from "@/components/FloatingShapes";

// API Base URL - defaults to localhost:8000 (FastAPI default port)
// To customize, set VITE_API_URL in your .env file
const API_BASE_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

interface PredictionResult {
  "predicted class": string;
  confidence: number;
}

const Detect = () => {
  const [image, setImage] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFile = useCallback(async (file: File) => {
    if (!file.type.startsWith("image/")) {
      setError("Please upload a valid image file");
      return;
    }

    const reader = new FileReader();
    reader.onload = async (e) => {
      setImage(e.target?.result as string);
      setResult(null);
      setError(null);
      setLoading(true);

      try {
        const formData = new FormData();
        formData.append("file", file);

        const response = await fetch(`${API_BASE_URL}/uploadfile/`, {
          method: "POST",
          body: formData,
          // Don't set Content-Type header - browser will set it automatically with boundary for FormData
        });

        if (!response.ok) {
          const errorText = await response.text().catch(() => "Unknown error");
          throw new Error(`Server error (${response.status}): ${errorText}`);
        }

        const data: PredictionResult = await response.json();
        setResult(data);
      } catch (err) {
        let errorMessage = "Failed to analyze image. Please try again.";

        if (err instanceof TypeError && err.message.includes("fetch")) {
          errorMessage = `Cannot connect to server at ${API_BASE_URL}. Please ensure:\n1. The FastAPI server is running\n2. CORS is enabled (see instructions below)`;
        } else if (err instanceof Error) {
          errorMessage = err.message;
        }

        setError(errorMessage);
        console.error("Error uploading file:", err);
      } finally {
        setLoading(false);
      }
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
  }, [handleFile]);

  const reset = () => {
    setImage(null);
    setResult(null);
    setLoading(false);
    setError(null);
  };

  const isHealthy = result ? result["predicted class"].toLowerCase().includes("healthy") : false;

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 pt-24 pb-16 relative">
        <FloatingShapes />
        <div className="container mx-auto px-4 relative z-10">
          <div className="text-center mb-10">
            <h1 className="text-3xl md:text-4xl font-bold mb-3 animate-fade-in">
              Plant Disease <span className="text-gradient-green">Detector</span>
            </h1>
            <p className="text-muted-foreground max-w-md mx-auto animate-fade-in" style={{ animationDelay: "0.1s" }}>
              Upload a photo of your plant leaf and our AI will analyze it for diseases
            </p>
          </div>

          <div className="max-w-xl mx-auto">
            {/* Upload Card */}
            {!image && (
              <div
                className={`glass rounded-2xl p-10 text-center animate-scale-in cursor-pointer transition-all duration-300 ${
                  dragOver ? "glow-green-lg border-primary/50 scale-[1.02]" : "hover:glow-green-sm"
                }`}
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-input")?.click()}
              >
                <div className="w-20 h-20 rounded-2xl bg-gradient-green mx-auto mb-6 flex items-center justify-center glow-green animate-float">
                  <Upload className="w-9 h-9 text-white" />
                </div>
                <h3 className="text-lg font-bold mb-2">Drag & Drop Your Image</h3>
                <p className="text-sm text-muted-foreground mb-6">or click to browse files</p>
                <Button className="bg-gradient-green text-white border-0 hover:opacity-90 hover:scale-105 transition-all duration-300 glow-green-sm">
                  <Upload className="w-4 h-4 mr-2" />
                  Upload Image
                </Button>
                <input
                  id="file-input"
                  type="file"
                  accept="image/*"
                  className="hidden"
                  onChange={(e) => {
                    const file = e.target.files?.[0];
                    if (file) handleFile(file);
                  }}
                />
              </div>
            )}

            {/* Image Preview + Loading/Result */}
            {image && (
              <div className="space-y-6 animate-fade-in">
                {/* Preview */}
                <div className="glass rounded-2xl p-4 overflow-hidden">
                  <img
                    src={image}
                    alt="Uploaded plant"
                    className="w-full h-64 object-cover rounded-xl"
                  />
                </div>

                {/* Loading */}
                {loading && (
                  <div className="glass rounded-2xl p-10 text-center animate-scale-in">
                    <div className="w-16 h-16 rounded-full bg-gradient-green mx-auto mb-5 flex items-center justify-center animate-spin-slow glow-green">
                      <Loader2 className="w-7 h-7 text-white animate-spin" />
                    </div>
                    <h3 className="text-lg font-bold mb-1">Analyzing Your Plant...</h3>
                    <p className="text-sm text-muted-foreground">AI is examining the image for diseases</p>
                  </div>
                )}

                {/* Error */}
                {error && !loading && (
                  <div className="glass rounded-2xl p-8 text-center animate-fade-in-up border-2 border-destructive/40 bg-card">
                    <div className="w-12 h-12 rounded-xl bg-destructive/15 mx-auto mb-4 flex items-center justify-center">
                      <AlertTriangle className="w-6 h-6 text-destructive" />
                    </div>
                    <h3 className="text-lg font-bold mb-2">Error</h3>
                    <p className="text-sm text-muted-foreground mb-6">{error}</p>
                    <Button
                      variant="outline"
                      onClick={reset}
                      className="hover:bg-primary/10 hover:border-primary/30 transition-all"
                    >
                      <RotateCcw className="w-4 h-4 mr-2" />
                      Try Again
                    </Button>
                  </div>
                )}

                {/* Result */}
                {result && !loading && !error && (
                  <div
                    className={`rounded-2xl p-8 animate-fade-in-up border-2 ${
                      isHealthy
                        ? "border-primary/40 glow-green bg-card"
                        : "border-accent/40 bg-card"
                    }`}
                    style={{ boxShadow: isHealthy ? undefined : "0 0 20px hsl(48 90% 60% / 0.2)" }}
                  >
                    <div className="flex items-center gap-3 mb-4">
                      <div className={`w-12 h-12 rounded-xl flex items-center justify-center ${
                        isHealthy ? "bg-primary/15" : "bg-accent/15"
                      }`}>
                        {isHealthy ? (
                          <CheckCircle2 className="w-6 h-6 text-primary" />
                        ) : (
                          <AlertTriangle className="w-6 h-6 text-accent-foreground" />
                        )}
                      </div>
                      <div>
                        <h3 className="text-xl font-bold">{result["predicted class"]}</h3>
                        <p className="text-sm text-muted-foreground">Confidence: {result.confidence.toFixed(1)}%</p>
                      </div>
                    </div>

                    <p className="text-sm text-muted-foreground mb-5">
                      {isHealthy
                        ? "No signs of disease detected. Your plant appears to be in excellent health!"
                        : "Disease detected. Please consult with a plant health expert for treatment recommendations."}
                    </p>

                    <Button
                      variant="outline"
                      onClick={reset}
                      className="hover:bg-primary/10 hover:border-primary/30 transition-all"
                    >
                      <RotateCcw className="w-4 h-4 mr-2" />
                      Scan Another Plant
                    </Button>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Detect;