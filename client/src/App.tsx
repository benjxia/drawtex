import React, {Component} from 'react';
import './App.css';
import CanvasDraw from 'react-canvas-draw';

interface App_Props {

}
export default class App extends Component {
  canvas: CanvasDraw | null;
  constructor(props: App_Props) {
    super(props);
    this.canvas = null;
  }

  hello(): void {
    console.log("hi");
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <p>
            Edit <code>src/App.tsx</code> and save to reload.
            <br></br>
            <CanvasDraw ref= {CanvasDraw => (this.canvas = CanvasDraw)} 
                        onChange={() => this.hello()}/>
          </p>
        </header>
      </div>
    );
  }
}
