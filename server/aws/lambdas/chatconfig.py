import dataclasses
import enum

class RetrieverStrategyEnum(enum.Enum):
    FullDocStrategy = 1
    PreAndPostChunkStrategy = 2
    
@dataclasses.dataclass
class ChatConfiguration:
    print_trace:bool
    use_ivfadc:bool
    retreiver_strategy:RetrieverStrategyEnum

