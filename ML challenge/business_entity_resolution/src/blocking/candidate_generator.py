"""
Unified multi-pass candidate generator.
Combines exact name, token, address, and lexical retrieval blocks (B1 to B9).
Provides candidate deduplication, union, and recall auditing.
"""

from typing import Dict, List, Set, Tuple, Any, Optional
from collections import defaultdict
from .exact_name import ExactNameBlocker
from .token_name import TokenNameBlocker
from .address_keys import AddressKeysBlocker
try:
    from ..config import BLOCKING_CONFIG
except (ImportError, ValueError):
    from config import BLOCKING_CONFIG

class MultiPassCandidateGenerator:
    def __init__(self, max_candidates: int = BLOCKING_CONFIG["max_candidates_per_entity"]):
        self.max_candidates = max_candidates
        self.exact_name_blocker = ExactNameBlocker(
            prefix_len=BLOCKING_CONFIG["min_name_prefix_len"],
            max_bucket_size=BLOCKING_CONFIG["max_block_bucket_size"]
        )
        self.token_name_blocker = TokenNameBlocker(
            max_bucket_size=BLOCKING_CONFIG["max_block_bucket_size"]
        )
        self.address_blocker = AddressKeysBlocker(
            max_bucket_size=BLOCKING_CONFIG["max_block_bucket_size"]
        )

    def index_target_record(
        self,
        entity_id: str,
        norm_name: Dict[str, Any],
        norm_addr: Dict[str, Any],
        addr_comp: Dict[str, Any],
    ):
        """Add a target record (S2 or S3) to all blocking indexes."""
        # B1 & B2: Exact & Prefix Name
        self.exact_name_blocker.add_record(
            entity_id=entity_id,
            legal_name=norm_name.get("legal", ""),
            compact_name=norm_name.get("compact", ""),
        )

        # B3: Token Name
        self.token_name_blocker.add_record(
            entity_id=entity_id,
            tokens=norm_name.get("tokens", []),
        )

        # B4, B5, B6: Address Keys
        first_name_token = norm_name.get("tokens", [""])[0] if norm_name.get("tokens") else ""
        first_addr_token = norm_addr.get("tokens", [""])[0] if norm_addr.get("tokens") else ""
        self.address_blocker.add_record(
            entity_id=entity_id,
            postal_code=addr_comp.get("postal_code"),
            state=addr_comp.get("state"),
            first_name_token=first_name_token,
            house_num=addr_comp.get("house_number"),
            first_addr_token=first_addr_token,
        )

    def generate_candidates_for_query(
        self,
        s1_id: str,
        norm_name: Dict[str, Any],
        norm_addr: Dict[str, Any],
        addr_comp: Dict[str, Any],
        lexical_candidates: Optional[List[str]] = None,
    ) -> List[str]:
        """Union candidates across all blocking rules, prioritize by rule specificity, and deduplicate."""
        cands: Set[str] = set()

        # Pass 1: Exact legal name (Highest precision)
        b1_b2 = self.exact_name_blocker.get_candidates(
            legal_name=norm_name.get("legal", ""),
            compact_name=norm_name.get("compact", ""),
        )
        cands.update(b1_b2)

        # Pass 2: Distinctive token name
        b3 = self.token_name_blocker.get_candidates(tokens=norm_name.get("tokens", []))
        cands.update(b3)

        # Pass 3: Address structured keys
        first_name_token = norm_name.get("tokens", [""])[0] if norm_name.get("tokens") else ""
        first_addr_token = norm_addr.get("tokens", [""])[0] if norm_addr.get("tokens") else ""
        b4_b6 = self.address_blocker.get_candidates(
            postal_code=addr_comp.get("postal_code"),
            state=addr_comp.get("state"),
            first_name_token=first_name_token,
            house_num=addr_comp.get("house_number"),
            first_addr_token=first_addr_token,
        )
        cands.update(b4_b6)

        # Pass 4: Lexical / TF-IDF candidates
        if lexical_candidates:
            cands.update(lexical_candidates)

        # Deduplicate and bound
        cand_list = list(cands)
        if len(cand_list) > self.max_candidates:
            cand_list = cand_list[:self.max_candidates]

        return cand_list

    @staticmethod
    def compute_blocking_recall(
        ground_truth: Dict[str, Set[str]],
        generated_candidates: Dict[str, List[str]]
    ) -> Dict[str, float]:
        """Evaluate blocking recall ceiling: recovered_true / total_true."""
        total_true_pairs = 0
        recovered_true_pairs = 0
        total_candidates = 0

        for s1_id, true_matches in ground_truth.items():
            if not true_matches:
                continue
            total_true_pairs += len(true_matches)
            cands = set(generated_candidates.get(s1_id, []))
            total_candidates += len(cands)
            recovered_true_pairs += len(true_matches & cands)

        recall = (recovered_true_pairs / total_true_pairs) if total_true_pairs > 0 else 1.0
        avg_cands = (total_candidates / len(ground_truth)) if ground_truth else 0.0

        return {
            "blocking_recall": round(recall, 5),
            "total_true_pairs": total_true_pairs,
            "recovered_true_pairs": recovered_true_pairs,
            "avg_candidates_per_entity": round(avg_cands, 2),
        }
