const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");
const app = express();
const fs = require("fs");
app.use(bodyParser.urlencoded({ extended: true }));
app.use(bodyParser.json());

let port = process.env.PORT;
if (port == null || port == "") {
  port = 3000;
}
app.listen(port, function () {
  console.log(`Server has started at port ${port}`);
});

app.post("/readPython", (req, res) => {
  var dataToSend;
  const python = spawn("python", ["Generate_Dataset.py", "abc", 1]);
  python.stdout.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on("exit", (code) => {
    console.log(`child process exited with ${code}, ${dataToSend}`);
  });
});

app.post("/readPython2", (req, res) => {
  var dataToSend;
  const python = spawn("python", ["Recognizer.py", 0]);
  python.stdout.on("data", (data) => {
    console.error(`stderr: ${data}`);
  });

  python.on("exit", (code) => {
    console.log(`child process exited with ${code}, ${dataToSend}`);
  });
});
