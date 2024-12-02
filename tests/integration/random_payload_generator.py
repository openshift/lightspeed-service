# Copyright © 2019, 2020, 2024 Pavel Tisnovsky, Red Hat
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Generator of random payload for testing REST API, configuration loaders, and test frameworks."""

import random
import string


class RandomPayloadGenerator:
    """Generator of random payload for testing API."""

    def __init__(
        self,
        max_iteration_deep=2,
        max_dict_key_length=5,
        max_list_length=5,
        max_string_length=10,
    ):
        """Initialize the random payload generator."""
        # Parameters that affect payload generator behaviour.
        self.iteration_deep = 0
        self.max_iteration_deep = max_iteration_deep
        self.max_list_length = max_list_length
        self.max_dict_key_length = max_dict_key_length
        self.max_string_length = max_string_length

        # Let's be strict which characters should be used as keys in generated
        # dictionaries.
        self.dict_key_characters = string.ascii_lowercase + string.ascii_uppercase + "_"

        # Dtto for string contents.
        self.string_value_characters = (
            string.ascii_lowercase
            + string.ascii_uppercase
            + "_"
            + string.punctuation
            + " "
        )

    def generate_random_string(self, n, uppercase=False, punctuations=False):
        """Generate random string of length=n."""
        # String prefix has limited character range.
        prefix = random.choice(string.ascii_lowercase)  # noqa: S311

        # The rest of string is more relaxed and the set of allowed characters
        # is configurable.
        mix = string.ascii_lowercase + string.digits

        if uppercase:
            mix += string.ascii_uppercase
        if punctuations:
            mix += string.punctuation

        suffix = "".join(random.choice(mix) for _ in range(n - 1))  # noqa: S311
        return prefix + suffix

    def generate_random_key_for_dict(self, data):
        """Generate a string key to be used in dictionary."""
        existing_keys = data.keys()
        # Try to generate new key which needs to be unique, i.e. it can not be
        # already contained in set of keys.
        while True:
            new_key = self.generate_random_string(self.max_string_length)
            if new_key not in existing_keys:
                return new_key

    def generate_random_list(self, n):
        """Generate list filled in with random values."""
        # Generate random list which is to be heterogenous - it's items can
        # have (almost) any data type.
        return [
            self.generate_random_payload((int, str, float, bool, list, dict))
            for i in range(n)
        ]

    def generate_random_dict(self, n):
        """Generate dictionary filled in with random values."""
        # Generate set of random keys and list of random values. Set and list
        # should have the same number of items. Then construct dictionary by
        # zipping these two data structures.
        dict_content = (int, str, list, dict)
        return {
            self.generate_random_string(
                self.max_string_length
            ): self.generate_random_payload(dict_content)
            for i in range(n)
        }

    def generate_random_list_or_string(self):
        """Generate list filled in with random strings."""
        # List are recursive structure that can contain other lists or
        # dictionaries. So we need to check if we already reached recursion
        # limit or not.
        if self.iteration_deep < self.max_iteration_deep:
            self.iteration_deep += 1
            value = self.generate_random_list(self.max_list_length)
            self.iteration_deep -= 1
        else:
            value = self.generate_random_value(str)
        return value

    def generate_random_dict_or_string(self):
        """Generate dict filled in with random strings."""
        # Dictionaries are recursive structure that can contain other lists or
        # dictionaries. So we need to check if we already reached recursion
        # limit or not.
        if self.iteration_deep < self.max_iteration_deep:
            self.iteration_deep += 1
            value = self.generate_random_dict(self.max_dict_key_length)
            self.iteration_deep -= 1
        else:
            value = self.generate_random_value(str)
        return value

    def generate_random_value(self, data_type):
        """Generate one random value of given type."""
        # Create generators for all possible data types. Each generator is able
        # to generate random values of given data type.
        generators = {
            str: lambda: self.generate_random_string(
                self.max_string_length, uppercase=True, punctuations=True
            ),
            int: lambda: random.randrange(100000),  # noqa: S311
            float: lambda: random.random() * 100000.0,  # noqa: S311
            bool: lambda: bool(random.getrandbits(1)),
            list: self.generate_random_list_or_string,
            dict: self.generate_random_dict_or_string,
        }

        # Select the generator specified by `data_type` argument.
        generator = generators[data_type]

        # And simply return it.
        return generator()

    def generate_random_payload(self, restrict_types=None):
        """Generate random payload with possibly restricted data types."""
        # Types can be restricted by user during initialization of this class.
        # If types are not restricted, we use the default set - strings,
        # integers, floats, lists (recursive), dictionary (recursive), and
        # booleans.
        if restrict_types:
            types = restrict_types
        else:
            types = (str, int, float, list, dict, bool)

        selected_type = random.choice(types)  # noqa: S311

        return self.generate_random_value(selected_type)
