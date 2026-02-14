import { useEffect, useRef } from "react";
import { Link } from "react-router-dom";
import { Upload, Cpu, ShieldCheck, Zap, Target, Sprout } from "lucide-react";
import { Button } from "@/components/ui/button";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FloatingShapes from "@/components/FloatingShapes";

const useScrollReveal = () => {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((e) => {
          if (e.isIntersecting) {
            e.target.classList.add("opacity-100", "translate-y-0");
            e.target.classList.remove("opacity-0", "translate-y-8");
          }
        });
      },
      { threshold: 0.1 }
    );
    const children = ref.current?.querySelectorAll(".reveal");
    children?.forEach((c) => observer.observe(c));
    return () => observer.disconnect();
  }, []);
  return ref;
};

const steps = [
  { icon: Upload, title: "Upload Image", desc: "Take a photo or upload an image of your plant leaf." },
  { icon: Cpu, title: "AI Analysis", desc: "Our AI model analyzes the image for signs of disease." },
  { icon: ShieldCheck, title: "Get Results", desc: "Receive a detailed diagnosis with care recommendations." },
];

const features = [
  { icon: Zap, title: "Instant Results", desc: "Get disease diagnosis in seconds, not days." },
  { icon: Target, title: "95%+ Accuracy", desc: "Trained on millions of plant images for precision." },
  { icon: Sprout, title: "50+ Crops", desc: "Supports a wide variety of crops and plant species." },
];

const Index = () => {
  const scrollRef = useScrollReveal();

  return (
    <div className="min-h-screen" ref={scrollRef}>
      <Navbar />

      {/* Hero */}
      <section className="relative min-h-screen flex items-center justify-center overflow-hidden">
        <div className="absolute inset-0 bg-gradient-hero animate-gradient-shift" />
        <FloatingShapes />
        <div className="relative z-10 container mx-auto px-4 text-center">
          <h1 className="text-5xl md:text-7xl font-black text-white mb-6 leading-tight animate-fade-in">
            Detect Plant Diseases
            <br />
            <span className="text-accent">Instantly with AI</span>
          </h1>
          <p className="text-lg md:text-xl text-white/80 max-w-2xl mx-auto mb-10 animate-fade-in" style={{ animationDelay: "0.2s" }}>
            Upload a photo of your crop and get AI-powered diagnosis in seconds. Protect your harvest with early disease detection.
          </p>
          <Link to="/detect">
            <Button
              size="lg"
              className="bg-white text-forest font-bold text-lg px-10 py-6 rounded-xl hover:scale-105 transition-all duration-300 animate-pulse-glow animate-fade-in"
              style={{ animationDelay: "0.4s" }}
            >
              <Sprout className="w-5 h-5 mr-2" />
              Get Started
            </Button>
          </Link>
        </div>
        {/* Bottom fade */}
        <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-background to-transparent" />
      </section>

      {/* How It Works */}
      <section className="py-24 relative">
        <FloatingShapes />
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="reveal opacity-0 translate-y-8 transition-all duration-700 text-3xl md:text-4xl font-bold text-center mb-4">
            How It <span className="text-gradient-green">Works</span>
          </h2>
          <p className="reveal opacity-0 translate-y-8 transition-all duration-700 text-muted-foreground text-center mb-16 max-w-lg mx-auto">
            Three simple steps to diagnose your plant's health
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {steps.map((step, i) => (
              <div
                key={step.title}
                className="reveal opacity-0 translate-y-8 transition-all duration-700 glass rounded-2xl p-8 text-center hover:scale-105 hover:glow-green-sm group"
                style={{ transitionDelay: `${i * 150}ms` }}
              >
                <div className="w-16 h-16 rounded-2xl bg-gradient-green mx-auto mb-5 flex items-center justify-center glow-green-sm group-hover:scale-110 transition-transform duration-300">
                  <step.icon className="w-7 h-7 text-white" />
                </div>
                <div className="text-xs font-bold text-primary mb-2">STEP {i + 1}</div>
                <h3 className="text-lg font-bold mb-2">{step.title}</h3>
                <p className="text-sm text-muted-foreground">{step.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-24 bg-muted/30 relative">
        <div className="container mx-auto px-4 relative z-10">
          <h2 className="reveal opacity-0 translate-y-8 transition-all duration-700 text-3xl md:text-4xl font-bold text-center mb-4">
            Why <span className="text-gradient-green">PlantGuard</span>?
          </h2>
          <p className="reveal opacity-0 translate-y-8 transition-all duration-700 text-muted-foreground text-center mb-16 max-w-lg mx-auto">
            Fast, accurate, and built for modern agriculture
          </p>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 max-w-4xl mx-auto">
            {features.map((feat, i) => (
              <div
                key={feat.title}
                className="reveal opacity-0 translate-y-8 transition-all duration-700 rounded-2xl border border-border bg-card p-8 text-center hover:border-primary/30 hover:glow-green-sm transition-all group"
                style={{ transitionDelay: `${i * 150}ms` }}
              >
                <div className="w-14 h-14 rounded-xl bg-primary/10 mx-auto mb-5 flex items-center justify-center group-hover:bg-primary/20 transition-colors">
                  <feat.icon className="w-6 h-6 text-primary" />
                </div>
                <h3 className="text-lg font-bold mb-2">{feat.title}</h3>
                <p className="text-sm text-muted-foreground">{feat.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-hero opacity-90" />
        <FloatingShapes />
        <div className="relative z-10 container mx-auto px-4 text-center">
          <h2 className="reveal opacity-0 translate-y-8 transition-all duration-700 text-3xl md:text-4xl font-bold text-white mb-4">
            Ready to Protect Your Crops?
          </h2>
          <p className="reveal opacity-0 translate-y-8 transition-all duration-700 text-white/70 mb-10 max-w-md mx-auto">
            Start using PlantGuard today — it's fast, free, and incredibly accurate.
          </p>
          <Link to="/detect">
            <Button
              size="lg"
              className="reveal opacity-0 translate-y-8 transition-all duration-700 bg-white text-forest font-bold text-lg px-10 py-6 rounded-xl hover:scale-105"
            >
              Try It Now
            </Button>
          </Link>
        </div>
      </section>

      <Footer />
    </div>
  );
};

export default Index;
