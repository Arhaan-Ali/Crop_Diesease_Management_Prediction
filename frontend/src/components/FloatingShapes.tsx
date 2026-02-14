const FloatingShapes = () => (
  <div className="absolute inset-0 overflow-hidden pointer-events-none">
    {/* Circles */}
    <div className="absolute top-20 left-[10%] w-72 h-72 rounded-full bg-primary/5 animate-float blur-xl" />
    <div className="absolute top-40 right-[15%] w-48 h-48 rounded-full bg-mint/10 animate-float-reverse blur-lg" />
    <div className="absolute bottom-32 left-[20%] w-56 h-56 rounded-full bg-leaf/8 animate-float-slow blur-xl" />
    <div className="absolute bottom-20 right-[10%] w-36 h-36 rounded-full bg-primary/8 animate-float blur-lg" />

    {/* Leaf-like shapes */}
    <div className="absolute top-[30%] left-[5%] w-16 h-16 rounded-full bg-primary/10 animate-float" style={{ animationDelay: "1s" }} />
    <div className="absolute top-[60%] right-[8%] w-12 h-12 rounded-full bg-mint/15 animate-float-reverse" style={{ animationDelay: "2s" }} />
    <div className="absolute top-[15%] right-[30%] w-20 h-20 rounded-full bg-leaf/10 animate-float-slow" style={{ animationDelay: "3s" }} />
  </div>
);

export default FloatingShapes;
