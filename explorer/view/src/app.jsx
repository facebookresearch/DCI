/*
 * Copyright (c) 2017-present, Facebook, Inc.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

import React from "react";
import ReactDOM from "react-dom";
import {Item} from "./components/view_component";

/* ================= Application Components ================= */

function MainApp() {
  const [loadedData, setLoadedData] = React.useState(null);

  React.useEffect(() => {
    const loc = window.location.pathname;
    const requestOptions = {
        method: 'GET'
    };
    fetch('/getdata' + loc, requestOptions)
        .then(response => response.json())
        .then(data => setLoadedData(data));

  // empty dependency array means this effect will only run once (like componentDidMount in classes)
  }, []);

  if (loadedData === null) {
    return <h1>Loading...</h1>;
  }

  const itemData = loadedData.item;
  const next = loadedData.next;
  const prev = loadedData.prev;
  const curr = loadedData.curr;

  return (
    <div>
      <p>
        <a href={prev}>[prev]</a> 
        <span>{curr}</span> 
        <a href={next}>[next]</a> 
      </p>
      <Item item={itemData}/>
    </div>
  );
}

ReactDOM.render(<MainApp />, document.getElementById("app"));
