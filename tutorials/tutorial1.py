# %%
# Markdown
"""
# Tutorial 1 — One Small Process, Big Payoffs

**What you'll build:** a tiny `ChatProcess` you can call with a string, a `Msg`, or a `ListDialog`. You'll plug it into OpenAI's Responses API, compose prompts from reusable parts (`ModuleList`), route across personas with a named map (`ModuleDict`), render readable snapshots, and quickly serialize state so you can **save/restore** behavior.

**Why Dachi?**

* **Composability**: Build systems from small, strongly typed pieces.
* **Reproducibility**: `spec()` & `state_dict()` mean you can checkpoint the *instructions* and the *structure* as your system evolves.
* **Simplicity**: One process class, progressively extended—no class-per-cell sprawl.
"""

# %%
# Python
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import InitVar
from dachi.core import (
    BaseModule, Param, Attr,
    ModuleList, ModuleDict,
    Msg, ListDialog,
    render
)
from dachi.proc import Process

# Define all our classes at the top of the notebook
class ChatProcess(Process):
    """A simple chat process that accepts str/Msg/ListDialog and returns a structured response"""
    
    system_prompt: InitVar[str] = "You are a helpful Dachi tutorial assistant."
    
    def __post_init__(self, system_prompt: str):
        super().__post_init__()
        # System prompt given at instantiation
        self.system_prompt = Param(system_prompt)
        
        # User-provided context that we'll embed in prompts
        self.material = Attr("")
        
        # Pluggable OpenAI Responses caller  
        self.caller = Attr(None)
    
    def _normalize(self, x: Union[str, Msg, List[Msg], ListDialog]) -> str:
        """Normalize input to a single content string"""
        if isinstance(x, str):
            return x
        elif isinstance(x, Msg):
            return x.text
        elif isinstance(x, ListDialog):
            # Combine all messages into a single string
            return "\n".join(f"{msg.role}: {msg.text}" for msg in x)
        elif isinstance(x, list) and all(isinstance(m, Msg) for m in x):
            # Handle List[Msg]
            return "\n".join(f"{msg.role}: {msg.text}" for msg in x)
        else:
            raise ValueError(f"Unsupported input type: {type(x)}")
    
    def _render_material(self) -> str:
        """Render material as a string"""
        if isinstance(self.material, dict):
            return "\n".join(f"{k}: {v}" for k, v in self.material.items())
        return str(self.material)
    
    def forward(self, x: Union[str, Msg, List[Msg], ListDialog]) -> Dict[str, Any]:
        """Process input and return {prompt, output}"""
        # Normalize input to string
        user_content = self._normalize(x)
        
        # Compose the prompt
        prompt_parts = [
            f"SYSTEM:\n{self.system_prompt}",
            f"\nMATERIAL:\n{self._render_material()}",
            f"\nUSER:\n{user_content}"
        ]
        prompt = "\n".join(prompt_parts)
        
        # Generate output
        if self.caller is None:
            # Offline echo mode
            output = f"[echo] {user_content}"
        else:
            # Call the external API
            output = self.caller(prompt)
        
        return {
            "prompt": prompt,
            "output": output
        }


class PromptPart(BaseModule):
    """A reusable prompt component"""
    
    text: InitVar[str] = ""
    
    def __post_init__(self, text: str):
        super().__post_init__()
        self.text = Param(text)


class PromptKit(BaseModule):
    """A collection of prompt parts using ModuleList"""
    
    parts: InitVar[ModuleList] = None
    
    def __post_init__(self, parts: Optional[ModuleList]):
        super().__post_init__()
        self.parts = parts or ModuleList(items=[])


class MultiChat(BaseModule):
    """Route requests through named ChatProcess instances"""
    
    bots: InitVar[ModuleDict] = None
    
    def __post_init__(self, bots: Optional[ModuleDict]):
        super().__post_init__()
        self.bots = bots or ModuleDict(items={})
    
    def forward(self, name: str, x: Union[str, Msg, List[Msg], ListDialog]) -> Dict[str, Any]:
        """Call the named bot"""
        if name not in self.bots:
            raise ValueError(f"Unknown bot: {name}. Available: {list(self.bots.keys())}")
        return self.bots[name](x)

# %%
# Markdown
"""
## 0) Setup & the Single Class We'll Reuse

We keep definitions in one place at the top:

* `ChatProcess` (the star of this tutorial)
  * `system_prompt: Param[str]` — given at instantiation
  * `material: Attr[str|dict]` — user-provided context we'll embed
  * `caller: Attr[Callable[[str], str]|None]` — pluggable OpenAI Responses caller
  * Input normalization: `str | Msg | List[Msg]` → one content string
  * Output: a small dict `{prompt, output}`

Later, we'll also **define once** (still in the same top cell):

* `PromptPart` + `PromptKit` (to show `ModuleList`)
* `MultiChat` (to show `ModuleDict` of named personas)

> Everything else below are **small cells that *use* these classes**.
"""

# %%
# Markdown
"""
## 1) Hello, `ChatProcess` (offline echo)

**Goal:** Prove the shape of the API before calling any external service.

**What you'll do**

* Instantiate with a **system message**.
* Call it with a simple string.
* See that `material` is empty and output is a local echo.

**What to run**
"""

# %%
# Python
cp = ChatProcess(system_prompt="You are a helpful Dachi tutorial assistant.")
cp("Hi Dachi!")

# %%
# Markdown
"""
## 2) Turn on OpenAI Responses (one small injection)

**Goal:** Use the **Responses API** without coupling our whole notebook to an SDK.

**What you'll do**

* Create a tiny function that calls OpenAI's Responses endpoint.
* Assign it to `cp.caller`.
* Call `cp(...)` again—now it returns model text.

**What to run**
"""

# %%
# Python
# Commented out for testing - uncomment when you have OpenAI API key
# from openai import OpenAI
# client = OpenAI()

# def responses_call(prompt: str) -> str:
#     r = client.responses.create(model="gpt-5", input=prompt)
#     return r.output_text

# cp.caller = responses_call
# cp("Give me one sentence on why composability matters.")

# For testing without OpenAI API key:
def mock_responses_call(prompt: str) -> str:
    return "Composability enables building complex AI systems from simple, reusable components."

cp.caller = mock_responses_call
cp("Give me one sentence on why composability matters.")

# %%
# Markdown
"""
**Why this design**

* The call is **isolated** behind `caller`. If APIs change or you switch vendors, you swap one function—**no rewrites** elsewhere.
"""

# %%
# Markdown
"""
## 3) Param vs Attr in action (without ML… yet)

**Goal:** Make `material` useful and show how it changes the prompt.

**What you'll do**

* Keep the same `ChatProcess`.
* Set `material` to a short dict.
* Call `cp(...)` again and see the system+material appear inside the prompt.

**What to run**
"""

# %%
# Python
cp.material = {
    "project": "Dachi",
    "goal": "simple, composable AI that stays maintainable",
    "tone": "practical and concise"
}
cp("Explain the benefits in 2 short bullets.")

# %%
# Markdown
"""
**Takeaway**

* `system_prompt` is a **Param**—part of the module's identity.
* `material` is an **Attr**—runtime context you can change freely.
* We'll use these views (Param vs Attr) to **save/restore** later. That's the foundation that will enable ML over instructions/structure in future tutorials.
"""

# %%
# Markdown
"""
## 4) Messages & Dialogs (one surface for multiple inputs)

**Goal:** Accept `str`, `Msg`, or `ListDialog` with zero extra code.

**What you'll do**

* Create messages and a short dialog.
* Call `cp(dialog)`—internally we'll normalize to a single content string for Responses.

**What to run**
"""

# %%
# Python
u1 = Msg(role="user", text="What is Dachi?")
a1 = Msg(role="assistant", text="A framework for composable AI.")
u2 = Msg(role="user", text="How do I save state?")

dialog = ListDialog(messages=[u1, a1, u2])
cp(dialog)

# %%
# Markdown
"""
**Why this matters**
One process surface keeps cohesion high. You don't need parallel functions for strings vs dialogs—**less drift, fewer bugs**.
"""

# %%
# Markdown
"""
## 5) A quick peek at serialization & `parameters()` (1 minute)

> We're not doing training or ML here; we just show how you'd **store and restore** behavior later.

**What you'll do**

* Compare `spec()` (Params only) with `state_dict()` (Params + Attr).
* List `parameters()` to see what's considered a Param.
* Rebuild a new `ChatProcess` from `spec()`.

**What to run**
"""

# %%
# Python
spec_view = cp.spec()            # Params only (e.g., system_prompt)
state_view = cp.state_dict()     # Params + Attr (includes material)
list(cp.parameters())            # iterate (name, Param) pairs

cp2 = ChatProcess.from_spec(spec_view)
cp2.material = cp.material       # copy over runtime state only if you want it
cp2("Confirm we restored behavior without copying code.")

# %%
# Markdown
"""
**Why this matters**
This is how you **save** system versions and **reproduce** results. In later tutorials we'll use these same hooks to **optimize instructions and structure** over time.
"""

# %%
# Markdown
"""
## 6) Reusable prompt parts with `ModuleList`

**Goal:** Move from ad-hoc strings to a structured set of reusable prompt parts.

**What you'll do**

* Build a `PromptKit` with a `ModuleList[PromptPart]`.
* Materialize `cp.material` from the kit, then call `cp(...)`.

**What to run**
"""

# %%
# Python
kit = PromptKit(parts=ModuleList(items=[
    PromptPart(text="You must be concise."),
    PromptPart(text="Answer with 2 bullets."),
    PromptPart(text="Avoid marketing jargon.")
]))

cp.material = "\n".join(p.text.data for p in kit.parts)
cp("Summarize Dachi's core benefits.")

# %%
# Markdown
"""
**Nice side effect**
`ModuleList` is **serializable**, so your prompt kit becomes a stable artifact—you can version it, share it, and swap parts confidently.
"""

# %%
# Markdown
"""
## 7) Named personas with `ModuleDict`

**Goal:** Route requests through a map of named `ChatProcess` instances, each with a different system message.

**What you'll do**

* Create two `ChatProcess` personas ("teacher" and "concise") and store them in a `ModuleDict`.
* Use `MultiChat` to call by name.

**What to run**
"""

# %%
# Python
teacher = ChatProcess(system_prompt="You are a patient teacher. Explain step by step.")
concise = ChatProcess(system_prompt="You answer in one crisp sentence.")
teacher.caller = concise.caller = mock_responses_call  # same injected caller

mc = MultiChat(bots=ModuleDict(items={"teacher": teacher, "concise": concise}))
mc.forward("teacher", "Explain Param vs Attr in Dachi.")
mc.forward("concise", "Explain Param vs Attr in Dachi.")

# %%
# Markdown
"""
**Why this matters**
`ModuleDict` gives you a **typed, serializable registry** of behaviors. You can checkpoint the whole map and bring it back exactly later.
"""

# %%
# Markdown
"""
## 8) Rendering readable snapshots

**Goal:** Produce human-readable views of specs, states, and summaries.

**What you'll do**

* Render `spec()` and `state_dict()` for both a single `ChatProcess` and the `MultiChat` container.

**What to run**
"""

# %%
# Python
from dachi.core import render

print(render(cp.spec()))
print(render(cp.state_dict()))
print(render(mc.spec()))

# %%
# Markdown
"""
**Why this matters**
When teams audit, share, and debug, **readable artifacts** are gold. Dachi emits compact, consistent representations.
"""

# %%
# Markdown
"""
## 9) Wrap-up: What you have now

* **One small process** with three inputs (string, message, dialog), one output, and a pluggable LLM caller.
* **Composable context** using `ModuleList` and **named personas** with `ModuleDict`.
* **Serialization & parameters()** glimpsed—enough to save/restore now, and to support learning in future tutorials.
* **Rendering** to produce stable, human-readable snapshots.

**Where we're going next**

* Add lightweight structure for multi-step processing.
* Use serialization for **controlled iteration** (tuning instructions safely).
* Optional: streaming and structured outputs.
"""

# %%
# Markdown
"""
### Appendix: What the top-cell class looks like (for reference)

We define these **once** at the top of the notebook:

* `ChatProcess(system_prompt: str)` with:
  * `system_prompt: Param[str]`
  * `material: Attr[str|dict]`
  * `caller: Attr[Callable[[str], str]|None]`
  * `_normalize`, `_render_material`, and `forward` (compose `SYSTEM/MATERIAL/USER` → call or echo)
* `PromptPart` and `PromptKit` to show `ModuleList`
* `MultiChat` with `bots: ModuleDict[str, ChatProcess]` and `forward(name, x)`

> With these in place, every section above is just a tiny usage cell—**one idea at a time**.
"""