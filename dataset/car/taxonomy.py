import copy
import math
import typing as t
from dataclasses import dataclass, field
from enum import Enum


class AttributeTypeTaxonomy(Enum):
    category = "category"
    number = "number"
    condition = "condition"


@dataclass
class ChoicesTaxonomy:
    values: t.List[str]

    def __iter__(self):
        for v in self.values:
            yield v

    def __eq__(self, other) -> bool:
        if isinstance(other, ChoicesTaxonomy):
            return self.values == other.values
        else:
            return False


@dataclass
class NumericalTaxonomy:
    min: float
    max: float
    step: float

    def to_dict(self) -> t.Dict[str, float]:
        return {
            "min": self.min,
            "max": self.max,
            "step": self.step,
        }

    def __iter__(self):
        for i in range(self.n_steps):
            yield str(self.min + i * self.step)

    @property
    def n_steps(self) -> int:
        return int((self.max - self.min) / self.step) + 1

    def __eq__(self, other) -> bool:
        if isinstance(other, NumericalTaxonomy):
            return (self.min == other.min) and (self.max == other.max) and (self.step == other.step)
        else:
            return False


@dataclass
class ConditionTaxonomy:
    values: t.List[str]
    taxonomy: str
    choices: ChoicesTaxonomy

    def to_dict(self) -> t.Dict[str, t.Union[str, t.Sequence[str]]]:
        return {
            "choices": self.values,
            "condition": {
                "taxonomy": self.taxonomy,
                "choices": self.choices.values,
            },
        }

    def __iter__(self):
        for v in self.values:
            yield v

    def __eq__(self, other) -> bool:
        if isinstance(other, ConditionTaxonomy):
            return (
                (self.values == other.values)
                and (self.taxonomy == other.taxonomy)
                and (self.choices == other.choices)
            )
        else:
            return False


@dataclass
class AttributeTaxonomy:
    name: str
    description: str
    value: t.Union[ChoicesTaxonomy, NumericalTaxonomy, ConditionTaxonomy]
    allow_multiple: bool = False
    meta: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    @property
    def type(self) -> AttributeTypeTaxonomy:
        if isinstance(self.value, ChoicesTaxonomy):
            return AttributeTypeTaxonomy.category
        elif isinstance(self.value, NumericalTaxonomy):
            return AttributeTypeTaxonomy.number
        elif isinstance(self.value, ConditionTaxonomy):
            return AttributeTypeTaxonomy.condition
        else:
            raise ValueError(f"value attribute has unknown type {type(self.value)}")

    def __post_init__(self):
        assert self.description != ""
        if (
            self.type is AttributeTypeTaxonomy.condition
            or self.type is AttributeTypeTaxonomy.category
        ):
            self.value.values.append("Unclear")

    def to_dict(self) -> t.Dict[str, t.Dict[str, t.Any]]:
        if self.type == AttributeTypeTaxonomy.category:
            return self.category_to_dict()
        elif self.type == AttributeTypeTaxonomy.number:
            return self.number_to_dict()
        elif self.type == AttributeTypeTaxonomy.condition:
            return self.condition_to_dict()
        else:
            raise NotImplementedError(f"This type is not supported {self.type}")

    def category_to_dict(
        self,
    ) -> t.Dict[str, t.Dict[str, t.Union[str, t.Sequence[str]]]]:
        return {
            self.name: {
                "type": self.type.value,
                "description": self.description,
                "choices": self.value.values,
                "allow_multiple": self.allow_multiple,
            },
        }

    def number_to_dict(self) -> t.Dict[str, t.Dict[str, t.Union[str, t.Sequence[str]]]]:
        return {
            self.name: {
                "type": self.type.value,
                "description": self.description,
                "min": self.value.min,
                "max": self.value.max,
                "step": self.value.step,
            },
        }

    def condition_to_dict(
        self,
    ) -> t.Dict[str, t.Dict[str, t.Union[str, t.Sequence[str]]]]:
        output_dict = {
            "type": "category",
            "description": self.description,
            "allow_multiple": self.allow_multiple,
        }
        output_dict.update(self.value.to_dict())
        return {self.name: output_dict}

    @property
    def n_values(self) -> int:
        return (
            self.value.n_steps
            if self.type is AttributeTypeTaxonomy.number
            else len(self.value.values)
        )

    def value_to_index(self, value: t.Union[float, str]) -> int:
        if self.type is AttributeTypeTaxonomy.number:
            if not isinstance(value, float):
                try:
                    value = float(value)
                except ValueError:
                    value = float(value.split(" to ")[0])
            return math.floor((value - self.value.min) / self.value.step)
        else:
            assert isinstance(value, str), type(value)
            return self.value.values.index(value)

    def index_to_value(self, index: int) -> str:
        if self.type is AttributeTypeTaxonomy.number:
            return f"{self.value.min + index * self.value.step} to {self.value.min + (index + 1) * self.value.step}"
        else:
            return self.value.values[index]

    def __eq__(self, other) -> bool:
        if isinstance(other, AttributeTaxonomy):
            return (
                (self.name == other.name)
                and (self.value == other.value)
                and (self.allow_multiple == other.allow_multiple)
            )
        else:
            return False


class AttributesList:
    _items: t.Dict[str, AttributeTaxonomy]

    def __init__(self, items: t.Sequence[AttributeTaxonomy]):
        self._items = {}
        for item in items:
            assert not self.has(item)
            self._items.update({item.name: item})

    def has(self, attribute: t.Union[AttributeTaxonomy, str]) -> bool:
        if isinstance(attribute, AttributeTaxonomy):
            attribute = attribute.name
        return attribute in self._items.keys()

    def __repr__(self):
        return f"AttributesList[{', '.join(self._items.keys())}]"

    def append(self, item: AttributeTaxonomy) -> t.NoReturn:
        assert not self.has(item)
        self._items.update({item.name: item})

    def __iter__(self) -> AttributeTaxonomy:
        for item in self._items.values():
            yield item

    def __getitem__(self, item: t.Union[str, int]) -> AttributeTaxonomy:
        if isinstance(item, str):
            return self._items[item]
        elif isinstance(item, int):
            return next(value for i, value in enumerate(self._items.values()) if i == item)

    def __len__(self) -> int:
        return len(self._items)

    @property
    def n_combinations(self) -> int:
        """Calculates the number of possible combinations of the attributes"""
        return math.prod(attribute.n_values for attribute in self)

    @property
    def vector_length(self) -> int:
        """Calculates the length of a vector that can be used to represent the attributes"""
        return sum(attribute.n_values for attribute in self)

    def __eq__(self, other) -> bool:
        if isinstance(other, AttributesList):
            for item_self, item_other in zip(self, other):
                if item_self != item_other:
                    return False
            return True
        else:
            return False


@dataclass
class CategoryTaxonomy:
    name: str
    description: str
    attributes: AttributesList
    meta: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    def __post_init__(self):
        assert self.description != ""

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            self.name: {
                "description": self.description,
                "attributes": [attribute.to_dict() for attribute in self.attributes],
            },
        }

    def __eq__(self, other) -> bool:
        if isinstance(other, CategoryTaxonomy):
            return (self.name == other.name) and (self.attributes == self.attributes)
        else:
            return False


@dataclass
class CompleteTaxonomy:
    things: t.Sequence[CategoryTaxonomy]
    stuff: t.Sequence[CategoryTaxonomy]
    meta: t.Dict[str, t.Any] = field(default_factory=lambda: {})

    def to_dict(self) -> t.Dict[str, t.Any]:
        return {
            "things": [thing.to_dict() for thing in self.things],
            "stuff": [stuff.to_dict() for stuff in self.stuff],
        }

    def __iter__(self) -> CategoryTaxonomy:
        for thing in self.things:
            yield thing
        for stuff in self.stuff:
            yield stuff

    def __getitem__(self, item: int):
        if item < len(self.things):
            return self.things[item]
        else:
            return self.stuff[item - len(self.things)]

    def is_sane(self) -> bool:
        for category in self:
            for attribute in category.attributes:
                if (
                    (
                        attribute.type is AttributeTypeTaxonomy.category
                        and not isinstance(attribute.value, ChoicesTaxonomy)
                    )
                    or (
                        attribute.type is AttributeTypeTaxonomy.number
                        and not isinstance(attribute.value, NumericalTaxonomy)
                    )
                    or (
                        attribute.type is AttributeTypeTaxonomy.condition
                        and not isinstance(attribute.value, ConditionTaxonomy)
                    )
                ):
                    return False
                if attribute.type is AttributeTypeTaxonomy.condition:
                    dependent_attribute_name = attribute.value.taxonomy
                    dependent_attribute_values = attribute.value.choices.values
                    for value in dependent_attribute_values:
                        assert value in category.attributes[dependent_attribute_name].value.values
        return True

    @staticmethod
    def load() -> "CompleteTaxonomy":
        return CompleteTaxonomy(
            things=[
                CategoryTaxonomy(
                    name="Mid-to-Large Vehicle",
                    description="This category contains all possible medium to large vehicles, which basically include all vehicles with 4 wheels. It does not include bicycles, motorcycles and any other small or portable vehicles.",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Vehicle Visibility",
                                description="How much of the vehicle is visible?",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "0% to 20% is visible",
                                        "21% to 40% is visible",
                                        "41% to 60% is visible",
                                        "60% to 80% is visible",
                                        "81% to 100% is visible",
                                    ],
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Form",
                                description="Vehicle Form Factor; On Rails includes Trams or Trains",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Sedan",
                                        "Hatchback",
                                        "SUV",
                                        "Sport",
                                        "Van",
                                        "Pickup",
                                        "Truck",
                                        "Trailer",
                                        "Bus",
                                        "School Bus",
                                        "On Rails",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is the vehicle towing or being towed",
                                description="Is it towing something? or is it being towed?",
                                value=ChoicesTaxonomy(values=["Towing", "Being Towed", "Neither"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Type",
                                description="What is this vehicle used for?",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Police",
                                        "Ambulance",
                                        "Fire",
                                        "Construction",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Light Status",
                                description="whether the vehicle's lights are on or off."
                                "If the vehicle is a school bus then the emergency option implies that the stop sign is on."
                                "Emergency indicate any special type of lights that the vehicle has, for example ambulances, police cars ... etc",
                                value=ChoicesTaxonomy(values=["On", "Off", "Emergency"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Status",
                                description="whether the vehicle is parked, stopped in the road or moving",
                                value=ChoicesTaxonomy(values=["Moving", "Stopped", "Parked"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Direction",
                                description="which part of the vehicle is facing the camera (if it is facing the camera with some angle, pick the one that closely agrees)",
                                value=ChoicesTaxonomy(
                                    values=["Front", "Back", "Left Side", "Right Side"]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Small Vehicle",
                    description="It includes the remaining small or portable vehicles such as bicycles, motorcycles, scooters, ... etc.",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Vehicle Type",
                                description='What is the type of the vehicle? "Other" may include things such as scooters.',
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Bicycle",
                                        "Motorcycle",
                                        "Float Drivable Surface",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Status",
                                description="whether the vehicle is parked, stopped in the road or moving",
                                value=ChoicesTaxonomy(values=["Moving", "Stopped", "Parked"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Has Rider",
                                description="whether there is a rider or not",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Direction",
                                description="which part of the vehicle is facing the camera (if it is facing the camera with some angle, pick the one that closely agrees)",
                                value=ChoicesTaxonomy(
                                    values=["Front", "Back", "Left Side", "Right Side"]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Vehicle Has Sidecar",
                                description="Has the Vehicle a Sidecar? some vehicles such as motorcycles may have sidecars. This can be used for bicycles as well with an additional cart or something else.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Pedestrian",
                    description="A human object in the image",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Visibility",
                                description="How much of the pedestrian is visible?",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "0% to 20% is visible",
                                        "21% to 40% is visible",
                                        "41% to 60% is visible",
                                        "60% to 80% is visible",
                                        "81% to 100% is visible",
                                    ],
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Age",
                                description="Age of the pedestrian",
                                value=ChoicesTaxonomy(values=["Adult", "Child"]),
                            ),
                            AttributeTaxonomy(
                                name="Pedestrian Type",
                                description="Is the pedestrian either a police officer / construction worker or neither of them",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Police Officer",
                                        "Construction Worker",
                                        "Neither",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Activity",
                                description="The posture of the person",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Sitting",
                                        "Lying Down",
                                        "Standing",
                                        "Walking",
                                        "Running",
                                        "Riding",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is Using Vehicle",
                                description="Is pedestrian using a vehicle? whether the pedestrian is using a bicycle, motorcycle, scooter, wheelchair or something else",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Pushing Or Dragging",
                                description="whether the pedestrian is pushing something in front of him/her or pulling something behind.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Carrying",
                                description="whether the pedestrian is carrying anything (a child, a backpack).",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Hearing Impaired",
                                description="Is pedestrian hearing impaired? not necessarily due to a disability, but could be due to wearing headphones or any item that may prevent the pedestrian from hearing the surrounding environment.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Blind",
                                description="Is pedestrian blind? may not be clear to you, choose the best of what you believe",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                            AttributeTaxonomy(
                                name="Is Disabled",
                                description="Is pedestrian disabled? any kind of disability except for being blind or hair impaired. Please do NOT use unclear unless it looks like there is a disability that is not clear to you. Otherwise, probably it will be No.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Traffic Light",
                    description="contains the whole traffic light object not just the lights",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Traffic Light Type",
                                description="What kind of traffic light",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Circle Lamp",
                                        "Forward Arrow (should be an actual arrow pointing forward, not a circle lamp)",
                                        "Right Arrow",
                                        "Left Arrow",
                                        "U-Turn",
                                        "Pedestrian",
                                        "Unknown",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Traffic Light Status",
                                description="Traffic light color? for some traffic lights such as pedestrian Green would represent pedestrians can cross and Red for no. Black means it is neither working nor functioning",
                                value=ChoicesTaxonomy(values=["Green", "Yellow", "Red", "Black"]),
                            ),
                            AttributeTaxonomy(
                                name="Flashing Traffic Light",
                                description="Flashing traffic light? whether the traffic lights are flashing or not. May not be clear, choose what you believe is correct.",
                                value=ChoicesTaxonomy(values=["Yes", "No"]),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Traffic Sign",
                    description="contains the main body of the traffic sign",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Is Electronic",
                                description="is it a fixed sign or an electronic one that can change later",
                                value=ChoicesTaxonomy(values=["Electronic", "Fixed"]),
                            ),
                            AttributeTaxonomy(
                                name="Traffic Sign Type",
                                description="Traffic sign type",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Stop",
                                        "Speed Limit",
                                        "Construction",
                                        "Caution",
                                        "No Stopping",
                                        "No Parking",
                                        "No Turn Right",
                                        "No Turn Left",
                                        "Wrong Way",
                                        "Do Not Enter",
                                        "One Way",
                                        "Barrier",
                                        "Advertisement or Informative",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Speed Limit Value",
                                description="Speed Limit Value? In case of a speed limit sign, what's the max value for speed? set only when the traffic sign is a speed limit, otherwise set to any value",
                                # type=AttributeTypeTaxonomy.condition,
                                value=NumericalTaxonomy(
                                    min=5,
                                    max=400,
                                    step=5,
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Road Lane",
                    description="Just the lanes of the road",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Road Lane Type",
                                description="what kind of lanes?",
                                value=ChoicesTaxonomy(values=["Solid", "Broken", "Other"]),
                            ),
                            AttributeTaxonomy(
                                name="Road Lane Color",
                                description="The color of the lane",
                                value=ChoicesTaxonomy(values=["White", "Yellow"]),
                            ),
                        ]
                    ),
                ),
            ],
            stuff=[
                CategoryTaxonomy(
                    name="Sky",
                    description="The sky including clouds/sun",
                    attributes=AttributesList([]),
                ),
                CategoryTaxonomy(
                    name="Sidewalk",
                    description="sidewalk excluding the road",
                    attributes=AttributesList([]),
                ),
                CategoryTaxonomy(
                    name="Construction",
                    description="any kind of human made objects",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Construction Type",
                                description="what kind of construction",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Building",
                                        "Wall",
                                        "Fence",
                                        "Bridge",
                                        "Tunnel",
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Vegetation",
                    description="vegetation above the level of the ground that may prohibit a vehicle from going in its direction",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Vegetation Type",
                                description="type of the vegetation",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Trees",
                                        "Hedges",
                                        "Small Bushes",
                                        "All other Kinds Of Vertical Vegetation",
                                        "Other",
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Movable Object",
                    description="any thing that may move later in time",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Movable Object Type",
                                description="the type of the movable object",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Traffic Cones",
                                        "Debris",
                                        "Barriers",
                                        "Push-able or Pull-able",
                                        "Animal",
                                        "Umbrella",
                                        "Other",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is Animal On Ground",
                                description="Is Animal on ground? either on ground or otherwise flying or being carried by someone.",
                                value=ConditionTaxonomy(
                                    values=["On Ground", "No"],
                                    taxonomy="Movable Object Type",
                                    choices=ChoicesTaxonomy(values=["Animal"]),
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Is Animal Moving by Itself",
                                description="Is the animal moving by itself? if being carried set to No",
                                value=ConditionTaxonomy(
                                    values=["Yes", "No"],
                                    taxonomy="Movable Object Type",
                                    choices=ChoicesTaxonomy(values=["Animal"]),
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Static Object",
                    description="any objects that probably will be there for a long time",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Static Object Type",
                                description="the type of the static object",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Bicycle Rack",
                                        "Pole",
                                        "Rail Track",
                                        "Trash (or anything that holds trash)",
                                        "Fence or Barrier",
                                        "Guard Rail",
                                        "Other",
                                    ]
                                ),
                            ),
                        ]
                    ),
                ),
                CategoryTaxonomy(
                    name="Ground",
                    description="any ground object such as road and parking spots",
                    attributes=AttributesList(
                        [
                            AttributeTaxonomy(
                                name="Ground Type",
                                description="the type of the ground? Terrain includes grass, soil and sand.",
                                value=ChoicesTaxonomy(
                                    values=[
                                        "Terrain",
                                        "Road",
                                        "Pedestrian Sidewalk",
                                        "Curb",
                                        "Parking Lots and Driveways",
                                    ]
                                ),
                            ),
                            AttributeTaxonomy(
                                name="Terrain Type",
                                description="Terrain Type? what type of terrain it is. Set only when the type is terrain, otherwise it doesn't matter",
                                value=ConditionTaxonomy(
                                    values=["Grass", "Soil", "Sand", "Other"],
                                    taxonomy="Ground Type",
                                    choices=ChoicesTaxonomy(values=["Terrain"]),
                                ),
                            ),
                        ]
                    ),
                ),
            ],
        )

    def fetch(self, **kwargs) -> t.Union["CategoryTaxonomy", "AttributeTaxonomy"]:
        assert len(kwargs) > 0, kwargs
        taxonomy = copy.deepcopy(self)
        if "category" in kwargs:
            category = kwargs["category"]
            things = [cat for cat in taxonomy.things if cat.name == category]
            stuff = [cat for cat in taxonomy.stuff if cat.name == category]
            if len(things) == 1 and len(stuff) != 1:
                output = things[0]
                output.meta.update({"Is Thing": True})
            elif len(stuff) == 1:
                output = stuff[0]
                output.meta.update({"Is Thing": False})
            else:
                raise ValueError(f"{category} doesn't exist in things nor stuff")

            if "attribute" in kwargs:
                attribute = kwargs["attribute"]
                attrs = [attr for attr in output.attributes if attr.name == attribute]
                assert (
                    len(attrs) == 1
                ), f"{output.attributes} doesn't have {attribute} of {category}"
                is_thing = output.meta["Is Thing"]
                output = attrs[0]
                output.meta.update({"Is Thing": is_thing, "category": category})

            return output
        else:
            raise NotImplementedError(kwargs)

    def __len__(self) -> int:
        return len(self.things) + len(self.stuff)


class CsTranslator:
    """
    Converts between Cityscapes way of labeling and our way of labeling
    """

    # _cs2ours:
    #   - keys are cityscapes classes, the values corresponds to our taxonomy
    #   - values is a string matching the category in our taxonomy
    _cs2ours = {
        "road": "Ground",
        "sidewalk": "Sidewalk",
        "parking": "Ground",
        "rail track": "Static Object",
        "person": "Pedestrian",
        "persongroup": "Pedestrian",
        "rider": "Pedestrian",
        "ridergroup": "Pedestrian",
        "car": "Mid-to-Large Vehicle",
        "cargroup": "Mid-to-Large Vehicle",
        "truck": "Mid-to-Large Vehicle",
        "bus": "Mid-to-Large Vehicle",
        "on rails": "Mid-to-Large Vehicle",
        "motorcycle": "Small Vehicle",
        "bicycle": "Small Vehicle",
        "bicyclegroup": "Small Vehicle",
        "motorcyclegroup": "Small Vehicle",
        "caravan": "Mid-to-Large Vehicle",
        "ego vehicle": "Mid-to-Large Vehicle",
        "trailer": "Mid-to-Large Vehicle",
        "building": "Construction",
        "wall": "Construction",
        "fence": "Construction",
        "guard rail": "Static Object",
        "bridge": "Construction",
        "tunnel": "Construction",
        "pole": "Static Object",
        "polegroup": "Static Object",
        "traffic sign": "Traffic Sign",
        "traffic light": "Traffic Light",
        "vegetation": "Vegetation",
        "terrain": "Ground",
        "sky": "Sky",
        "ground": "Ground",
        "dynamic": "Movable Object",
        "static": "Static Object",
        "out of roi": "Ground",
        "license plate": "Static Object",
        "rectification border": "Traffic Sign",
        "train": "Mid-to-Large Vehicle",
    }

    _skipped_categories = {  # key is the category and value is attributes map
        "sky": None,
        "sidewalk": None,
        "building": ["Construction", ("Construction Type", "Building")],
        "wall": ["Construction", ("Construction Type", "Wall")],
        "fence": ["Construction", ("Construction Type", "Fence")],
        "tunnel": ["Construction", ("Construction Type", "Tunnel")],
        "guard rail": ["Static Object", ("Static Object Type", "Guard Rail")],
        "bridge": ["Construction", ("Construction Type", "Bridge")],
        "rail track": ["Static Object", ("Static Object Type", "Rail Track")],
        "pole": ["Static Object", ("Static Object Type", "Pole")],
        "polegroup": ["Static Object", ("Static Object Type", "Pole")],
        "parking": ["Ground", ("Ground Type", "Parking Lots and Driveways")],
        "road": ["Ground", ("Ground Type", "Road")],
    }

    def __init__(self, sanity_check: bool = True):
        if sanity_check:
            self.is_sane()

    def is_sane(self):
        ours = CompleteTaxonomy.load()
        assert ours.is_sane()
        for k, v in self._cs2ours.items():
            ours.fetch(category=v)
        for category_cs, car_map in self._skipped_categories.items():
            assert category_cs in self._cs2ours.keys()
            if car_map is not None:
                item = ours.fetch(category=car_map[0])
                assert item.attributes.has(car_map[1][0])
                assert car_map[1][1] in item.attributes[car_map[1][0]].value.values

    def Cs2Ours(self, name: str):
        try:
            return self._cs2ours[name]
        except KeyError:
            logger.debug(f"Key {name} not found, replacing with unknown")
            return "unknown"

    def Ours2Cs(self, name: str):
        # TODO: implement this one as well, it may requires changing the structure as the mapping is many to one not one to one so far
        raise NotImplementedError()

    def isSkipped(self, name: str) -> bool:
        return name in self._skipped_categories.keys()

    @property
    def no_attributes_categories(self):
        return [key for key, value in self._skipped_categories.items() if value is not None]

    def generate_attributes_map(self, label: str) -> t.Dict[str, str]:
        assert toSkip(label)
        item = self._skipped_categories[label]
        return (
            {
                item[1][0]: item[1][1],
            }
            if item
            else {}
        )


TAXONOMY = CompleteTaxonomy.load()
CST = CsTranslator(sanity_check=True)
CSMap = CST.Cs2Ours
toSkip = CST.isSkipped
autoGenerateAttributes = CST.generate_attributes_map
