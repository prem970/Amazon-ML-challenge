"""
Address-based blocking keys (B4, B5, B6).
B4: Exact postal code (strong locality evidence).
B5: Locality/State + name token (locality + identity).
B6: House number + address token (address structure).
"""

from typing import Dict, List, Set, Optional
from collections import defaultdict

class AddressKeysBlocker:
    def __init__(self, max_bucket_size: int = 150):
        self.max_bucket_size = max_bucket_size
        self.postal_index = defaultdict(list)
        self.state_name_index = defaultdict(list)
        self.house_addr_index = defaultdict(list)

    def add_record(
        self,
        entity_id: str,
        postal_code: Optional[str],
        state: Optional[str],
        first_name_token: Optional[str],
        house_num: Optional[str],
        first_addr_token: Optional[str],
    ):
        # B4: Postal code
        if postal_code:
            key = postal_code.strip().lower()
            if len(self.postal_index[key]) < self.max_bucket_size:
                self.postal_index[key].append(entity_id)

        # B5: State + First Name Token
        if state and first_name_token and len(first_name_token) >= 3:
            key = f"{state.strip().lower()}_{first_name_token.strip().lower()}"
            if len(self.state_name_index[key]) < self.max_bucket_size:
                self.state_name_index[key].append(entity_id)

        # B6: House Number + Address Token
        if house_num and first_addr_token and len(first_addr_token) >= 3:
            key = f"{house_num.strip().lower()}_{first_addr_token.strip().lower()}"
            if len(self.house_addr_index[key]) < self.max_bucket_size:
                self.house_addr_index[key].append(entity_id)

    def get_candidates(
        self,
        postal_code: Optional[str],
        state: Optional[str],
        first_name_token: Optional[str],
        house_num: Optional[str],
        first_addr_token: Optional[str],
    ) -> Set[str]:
        candidates = set()
        if postal_code:
            key = postal_code.strip().lower()
            if key in self.postal_index:
                candidates.update(self.postal_index[key])

        if state and first_name_token and len(first_name_token) >= 3:
            key = f"{state.strip().lower()}_{first_name_token.strip().lower()}"
            if key in self.state_name_index:
                candidates.update(self.state_name_index[key])

        if house_num and first_addr_token and len(first_addr_token) >= 3:
            key = f"{house_num.strip().lower()}_{first_addr_token.strip().lower()}"
            if key in self.house_addr_index:
                candidates.update(self.house_addr_index[key])

        return candidates
