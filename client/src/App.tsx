import React from 'react';
import './App.css';
import CanvasDraw from "react-canvas-draw";

function App() {
  let i: JSX.Element = <CanvasDraw />
  
  return (
    <div className="App">
      <header className="App-header">
        <p>
          Edit <code>src/App.tsx</code> and save to reload.
          <br></br>
          {i}
        </p>
      </header>
    </div>
  );
}
// TODO: switch to es6?
export default App;
