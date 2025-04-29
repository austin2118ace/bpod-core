from collections import OrderedDict
from typing import Annotated

from annotated_types import IsAscii
from graphviz import Digraph
from pydantic import BaseModel, Field, validate_call

StateMachineName = Annotated[IsAscii[str], Field(min_length=1, default='State Machine')]
StateName = Annotated[IsAscii[str], Field(min_length=1)]
StateTimer = Annotated[float, Field(ge=0, allow_inf_nan=False, default=0.0)]
StateChangeConditions = Annotated[dict[str, StateName], Field(default_factory=dict)]
OutputActions = Annotated[dict[str, int], Field(default_factory=dict)]


class State(BaseModel):
    """
    Represents a state in the state machine.

    Attributes
    ----------
    name : StateName
        The name of the state.
    timer : StateTimer, optional
        The duration of the state in seconds. Defaults to 0.
    state_change_conditions : StateChangeConditions, optional
        A dictionary mapping conditions to target states for transitions. Defaults to an
        empty dictionary.
    output_actions : StateChangeConditions, optional
        A dictionary of actions to be executed during the state. Defaults to an empty
        dictionary.
    comment : str, optional
        An optional comment describing the state.
    """

    name: StateName
    timer: StateTimer = StateTimer()
    state_change_conditions: StateChangeConditions = StateChangeConditions()
    output_actions: StateChangeConditions = StateChangeConditions()
    comment: str = ''

    class Config:  # noqa: D106
        validate_assignment = True


class StateMachine(BaseModel):
    """
    Represents a state machine with a collection of states.

    Attributes
    ----------
    name : StateMachineName
        The name of the state machine.
    states : OrderedDict[StateName, State]
        An ordered dictionary of states in the state machine.
    """

    name: StateMachineName
    states: OrderedDict[StateName, State] = Field(default_factory=OrderedDict)

    class Config:  # noqa: D106
        validate_assignment = True

    @validate_call
    def add_state(
        self,
        name: StateName,
        timer: StateTimer,
        state_change_conditions: StateChangeConditions = None,
        output_actions: OutputActions = None,
        comment: str = '',
    ) -> None:
        if name in self.states:
            raise ValueError(f"A state named '{name}' is already registered")
        if state_change_conditions is None:
            state_change_conditions = StateChangeConditions()
        self.states[name] = State.construct(
            name=name,
            timer=timer,
            state_change_conditions=state_change_conditions,
            output_actions=output_actions,
            comment=comment,
        )

    @property
    def digraph(self) -> Digraph:
        """
        Returns a graphviz Digraph instance representing the state machine.

        The Digraph includes:

        - A point-shaped node representing the start of the state machine,
        - An optional 'exit' node if any state transitions to 'exit',
        - Record-like nodes for each state displaying state name, timer, comment and
          output actions, and
        - Edges representing state transitions based on conditions.

        Returns
        -------
        Digraph
            A graphviz Digraph instance representing the state machine.

        Notes
        -----
        This method depends on theGraphviz system libraries to be installed.
        See https://graphviz.readthedocs.io/en/stable/manual.html#installation
        """
        # Initialize the Digraph with the name of the state machine
        digraph = Digraph(self.name)

        # Return an empty Digraph if there are no states
        if len(self.states) == 0:
            return digraph

        # Add the start node represented by a point-shaped node
        digraph.node(name='', shape='point')
        digraph.edge('', next(iter(self.states.keys())))

        # Add an 'exit' node if any state transitions to 'exit'
        if 'exit' in [
            target
            for state in self.states.values()
            for target in state.state_change_conditions.values()
        ]:
            digraph.node(name='exit', label='<<b>exit</b>>', shape='plain')

        # Add nodes for each state
        for state in self.states.values():
            # Create table rows for the state's comment and output actions
            comment = (
                f'<TR><TD ALIGN="LEFT" COLSPAN="2" BGCOLOR="LIGHTBLUE">'
                f'<I>{state.comment}</I></TD></TR>'
                if len(state.comment) > 0
                else ''
            )
            actions = ''.join(
                f'<TR><TD ALIGN="LEFT">{k}</TD><TD ALIGN="RIGHT">{v}</TD></TR>'
                for k, v in state.output_actions.items()
            )

            # Create label for the state node with its name, timer, comment, and actions
            label = (
                f'<<TABLE BORDER="1" CELLBORDER="0" CELLSPACING="0" ALIGN="LEFT">'
                f'<TR><TD BGCOLOR="LIGHTBLUE" ALIGN="LEFT"><B>{state.name}  </B></TD>'
                f'<TD BGCOLOR="LIGHTBLUE" ALIGN="RIGHT">{state.timer:g} s</TD></TR>'
                f'{comment}{actions}</TABLE>>'
            )

            # Add the state node to the Digraph
            digraph.node(name=state.name, label=label, shape='none')

            # Add edges for state transitions based on conditions
            for condition, target_state in state.state_change_conditions.items():
                digraph.edge(state.name, target_state, label=condition)

        return digraph
