import { useState } from "react";
import { Send, CheckCircle2, Github, Twitter, Linkedin } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import Navbar from "@/components/Navbar";
import Footer from "@/components/Footer";
import FloatingShapes from "@/components/FloatingShapes";

const Contact = () => {
  const [submitted, setSubmitted] = useState(false);
  const [form, setForm] = useState({ name: "", email: "", message: "" });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name || !form.email || !form.message) return;
    setSubmitted(true);
    setTimeout(() => setSubmitted(false), 4000);
    setForm({ name: "", email: "", message: "" });
  };

  return (
    <div className="min-h-screen flex flex-col">
      <Navbar />

      <main className="flex-1 pt-24 pb-16 relative">
        <FloatingShapes />
        <div className="container mx-auto px-4 relative z-10">
          <div className="text-center mb-10">
            <h1 className="text-3xl md:text-4xl font-bold mb-3 animate-fade-in">
              Get In <span className="text-gradient-green">Touch</span>
            </h1>
            <p className="text-muted-foreground max-w-md mx-auto animate-fade-in" style={{ animationDelay: "0.1s" }}>
              Have questions or feedback? We'd love to hear from you
            </p>
          </div>

          <div className="max-w-lg mx-auto">
            <div className="glass rounded-2xl p-8 animate-scale-in">
              {submitted && (
                <div className="mb-6 p-4 rounded-xl bg-primary/10 border border-primary/20 flex items-center gap-3 animate-fade-in">
                  <CheckCircle2 className="w-5 h-5 text-primary shrink-0" />
                  <p className="text-sm font-medium text-primary">Message sent successfully! We'll get back to you soon.</p>
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-5">
                <div>
                  <label className="text-sm font-medium mb-1.5 block">Name</label>
                  <Input
                    value={form.name}
                    onChange={(e) => setForm({ ...form, name: e.target.value })}
                    placeholder="Your name"
                    className="bg-background/50 border-border focus:border-primary focus:ring-primary/20 transition-all"
                    required
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-1.5 block">Email</label>
                  <Input
                    type="email"
                    value={form.email}
                    onChange={(e) => setForm({ ...form, email: e.target.value })}
                    placeholder="you@example.com"
                    className="bg-background/50 border-border focus:border-primary focus:ring-primary/20 transition-all"
                    required
                  />
                </div>
                <div>
                  <label className="text-sm font-medium mb-1.5 block">Message</label>
                  <Textarea
                    value={form.message}
                    onChange={(e) => setForm({ ...form, message: e.target.value })}
                    placeholder="Tell us what's on your mind..."
                    rows={5}
                    className="bg-background/50 border-border focus:border-primary focus:ring-primary/20 transition-all resize-none"
                    required
                  />
                </div>
                <Button
                  type="submit"
                  className="w-full bg-gradient-green text-white border-0 hover:opacity-90 hover:scale-[1.02] transition-all duration-300 glow-green-sm"
                >
                  <Send className="w-4 h-4 mr-2" />
                  Send Message
                </Button>
              </form>

              <div className="mt-8 pt-6 border-t border-border">
                <p className="text-sm text-muted-foreground text-center mb-4">Or find us on social media</p>
                <div className="flex justify-center gap-3">
                  {[Github, Twitter, Linkedin].map((Icon, i) => (
                    <a
                      key={i}
                      href="#"
                      className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center text-primary hover:bg-primary hover:text-primary-foreground transition-all duration-300 hover:scale-110"
                    >
                      <Icon className="w-4 h-4" />
                    </a>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </main>

      <Footer />
    </div>
  );
};

export default Contact;
