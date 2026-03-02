import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
from interface.cli import ArkheCLI
from utils.visualizer import show_graph

class ArkheGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Arkhe(n) – Artificial Substrate Intelligence")
        self.root.geometry("800x600")
        self.cli = ArkheCLI()
        self._build_ui()

    def _build_ui(self):
        # Title
        title = tk.Label(self.root, text="Arkhe(n) ASI", font=("Helvetica", 16, "bold"))
        title.pack(pady=5)

        # Output area
        self.output = scrolledtext.ScrolledText(self.root, height=25, state='normal')
        self.output.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.output.insert(tk.END, self.cli.intro + "\n")

        # Input frame
        input_frame = tk.Frame(self.root)
        input_frame.pack(fill=tk.X, padx=10, pady=5)
        tk.Label(input_frame, text="Command:").pack(side=tk.LEFT)
        self.input_entry = tk.Entry(input_frame, width=50)
        self.input_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.input_entry.bind("<Return>", self.on_enter)

        # Button frame
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=5)
        tk.Button(btn_frame, text="Visualize", command=self.visualize).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Coherence", command=self.show_coherence).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Postulates", command=self.show_postulates).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Exit", command=self.root.quit).pack(side=tk.LEFT, padx=5)

    def on_enter(self, event):
        cmd = self.input_entry.get()
        self.input_entry.delete(0, tk.END)
        self.output.insert(tk.END, f"> {cmd}\n")
        self.output.see(tk.END)
        # Process command in a separate thread to avoid GUI freeze
        threading.Thread(target=self.process_command, args=(cmd,), daemon=True).start()

    def process_command(self, cmd):
        # We need to capture output – could redirect stdout, but for simplicity,
        # we'll implement a few basic commands directly.
        parts = cmd.split()
        if not parts:
            return
        command = parts[0].lower()
        if command == "ask":
            question = ' '.join(parts[1:])
            if question:
                from utils.translator import translate_query
                ans = translate_query(question, self.cli.h)
                self.output.after(0, lambda: self.output.insert(tk.END, ans + "\n"))
        elif command == "simulate":
            self.cli.do_simulate(' '.join(parts[1:]))
            # cli prints, but we can't capture easily; maybe use a queue.
            # For now, just show coherence.
            self.output.after(0, lambda: self.output.insert(tk.END, f"Coherence: {self.cli.h.total_coherence():.4f}\n"))
        elif command == "coherence":
            self.output.after(0, lambda: self.output.insert(tk.END, f"Total coherence: {self.cli.h.total_coherence():.4f}\n"))
        elif command == "postulates":
            import io
            from contextlib import redirect_stdout
            f = io.StringIO()
            with redirect_stdout(f):
                from domains.metaphysics import show_postulates
                show_postulates()
            self.output.after(0, lambda: self.output.insert(tk.END, f.getvalue() + "\n"))
        elif command == "visualize":
            self.visualize()
        elif command == "exit":
            self.root.quit()
        else:
            self.output.after(0, lambda: self.output.insert(tk.END, "Unknown command.\n"))
        self.output.after(0, lambda: self.output.see(tk.END))

    def visualize(self):
        show_graph(self.cli.h)

    def show_coherence(self):
        c = self.cli.h.total_coherence()
        messagebox.showinfo("Coherence", f"Total coherence: {c:.4f}")

    def show_postulates(self):
        from domains.metaphysics import POSTULATES
        msg = "\n".join(POSTULATES)
        messagebox.showinfo("Postulates", msg)

    def run(self):
        self.root.mainloop()
