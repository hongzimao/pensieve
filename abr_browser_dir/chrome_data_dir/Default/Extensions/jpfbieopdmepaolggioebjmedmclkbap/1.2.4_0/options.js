var enableOnStartElement = document.getElementById("enableOnStartElement");
// Saves options to localStorage.
function save_options() {
    var newValue = "off";
    if (enableOnStartElement.checked) {
        newValue = "on";
    }
    localStorage["turnOnByDefault"] = newValue;
}
// Restores select box state to saved value from localStorage.
function restore_options() {
    if (localStorage["turnOnByDefault"] === "on") {
        enableOnStartElement.checked = true;
    }
}

enableOnStartElement.addEventListener("change", save_options);
restore_options();