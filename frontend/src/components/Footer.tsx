import { Link } from "react-router-dom";
import { Leaf, Github, Twitter, Linkedin } from "lucide-react";

const Footer = () => (
  <footer className="border-t border-border bg-muted/30">
    <div className="container mx-auto px-4 py-12">
      <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
        {/* Brand */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-green flex items-center justify-center">
              <Leaf className="w-4 h-4 text-white" />
            </div>
            <span className="text-lg font-bold text-gradient-green">PlantGuard</span>
          </div>
          <p className="text-sm text-muted-foreground max-w-xs">
            AI-powered plant health detection. Protect your crops with instant disease diagnosis.
          </p>
        </div>

        {/* Links */}
        <div className="space-y-3">
          <h4 className="font-semibold text-sm text-foreground">Quick Links</h4>
          <div className="flex flex-col gap-2">
            {[{ label: "Home", path: "/" }, { label: "Detect Disease", path: "/detect" }, { label: "Contact Us", path: "/contact" }].map((l) => (
              <Link key={l.path} to={l.path} className="text-sm text-muted-foreground hover:text-primary transition-colors">
                {l.label}
              </Link>
            ))}
          </div>
        </div>

        {/* Social */}
        <div className="space-y-3">
          <h4 className="font-semibold text-sm text-foreground">Follow Us</h4>
          <div className="flex gap-3">
            {[Github, Twitter, Linkedin].map((Icon, i) => (
              <a
                key={i}
                href="#"
                className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center text-primary hover:bg-primary hover:text-primary-foreground transition-all duration-300 hover:scale-110 hover:glow-green-sm"
              >
                <Icon className="w-4 h-4" />
              </a>
            ))}
          </div>
        </div>
      </div>

      <div className="mt-10 pt-6 border-t border-border text-center">
        <p className="text-xs text-muted-foreground">© 2026 PlantGuard. Built with 🌿 for healthier crops.</p>
      </div>
    </div>
  </footer>
);

export default Footer;
