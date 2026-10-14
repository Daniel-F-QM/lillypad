"""The application shell: what happens between `python main.py` and a window.

Just `lillypad/app/main.py`, holding `main()`. It is its own module because
startup ORDER is the one thing that cannot live anywhere else — the theme has to
be installed before the first widget exists, and nothing inside the window can
arrange that for itself.

Deliberately no re-export here. `from lillypad.app import main` would bind the
FUNCTION onto the package as the attribute `main`, shadowing the MODULE of the
same name, so `import lillypad.app.main` and `from lillypad.app import main`
would hand back two different objects. Import it as
`from lillypad.app.main import main`.
"""
