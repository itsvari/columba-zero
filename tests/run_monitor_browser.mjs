import {spawn} from "node:child_process";
import {writeFile} from "node:fs/promises";
import {pathToFileURL} from "node:url";

const [browserPath, htmlPath, profilePath, screenshotPath] = process.argv.slice(2);
const browser = spawn(browserPath, [
  "--headless=new", "--disable-gpu", "--disable-extensions", "--no-first-run",
  "--no-default-browser-check", "--disable-background-networking",
  `--user-data-dir=${profilePath}`, "--remote-debugging-pipe", "about:blank"
], {stdio: ["ignore", "ignore", "pipe", "pipe", "pipe"]});
const pending = new Map();
const subscriptions = new Map();
let requestNumber = 0;
let buffer = "";
let browserErrors = "";
const maximumErrorCharacters = 4000;
browser.stderr.on("data", chunk => {
  browserErrors = (browserErrors + chunk.toString()).slice(-maximumErrorCharacters);
});
browser.stdio[4].on("data", chunk => {
  buffer += chunk.toString();
  let delimiter = buffer.indexOf("\0");
  while (delimiter !== -1) {
    const message = JSON.parse(buffer.slice(0, delimiter));
    buffer = buffer.slice(delimiter + 1);
    const callback = pending.get(message.id);
    if (callback) {
      pending.delete(message.id);
      if (message.error) callback.reject(new Error(JSON.stringify(message.error)));
      else callback.resolve(message.result);
    }
    const eventKey = `${message.sessionId}:${message.method}`;
    const subscriber = subscriptions.get(eventKey);
    if (subscriber) {
      subscriptions.delete(eventKey);
      subscriber(message.params);
    }
    delimiter = buffer.indexOf("\0");
  }
});
browser.on("error", error => {
  for (const callback of pending.values()) callback.reject(error);
});
browser.on("exit", code => {
  for (const callback of pending.values()) {
    callback.reject(new Error(`Browser exited ${code}: ${browserErrors}`));
  }
});

function command(method, parameters = {}, sessionId = undefined) {
  requestNumber += 1;
  const id = requestNumber;
  return new Promise((resolve, reject) => {
    pending.set(id, {resolve, reject});
    browser.stdio[3].write(JSON.stringify({id, method, params: parameters, sessionId}) + "\0");
  });
}

const timeoutMs = 30_000;
const timeout = setTimeout(() => {
  process.stderr.write(`Browser test timeout: ${browserErrors}\n`);
  browser.kill("SIGKILL");
  process.exitCode = 1;
}, timeoutMs);
try {
  const {targetId} = await command("Target.createTarget", {url: "about:blank"});
  const {sessionId} = await command("Target.attachToTarget", {targetId, flatten: true});
  await command("Page.enable", {}, sessionId);
  await command("Emulation.setDeviceMetricsOverride", {
    width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false
  }, sessionId);
  const loaded = new Promise(resolve => {
    subscriptions.set(`${sessionId}:Page.loadEventFired`, resolve);
  });
  await command("Page.navigate", {url: pathToFileURL(htmlPath).href}, sessionId);
  await loaded;
  const result = await command("Runtime.evaluate", {
    expression: "window.monitorTestCompletion", awaitPromise: true, returnByValue: true
  }, sessionId);
  if (result.exceptionDetails || !result.result.value) {
    throw new Error(JSON.stringify(result));
  }
  if (screenshotPath) {
    const screenshot = await command("Page.captureScreenshot", {format: "png"}, sessionId);
    await writeFile(screenshotPath, Buffer.from(screenshot.data, "base64"));
  }
  process.stdout.write(JSON.stringify(result.result.value) + "\n");
} catch (error) {
  process.stderr.write(`${String(error)}\n${browserErrors}\n`);
  process.exitCode = 1;
} finally {
  clearTimeout(timeout);
  browser.kill("SIGTERM");
}
