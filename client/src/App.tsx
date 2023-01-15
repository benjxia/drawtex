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

  /**
   * Records the current state of the canvas as a URL
   * before feeding it into the model
   */
  updatePred(): void {
    // @ts-ignore: Unreachable code error
    let img_url: string = this.canvas?.getDataURL();
  }

  render() {
    return (
      <div className="App">
        <header className="App-header">
          <p>
            Edit <code>src/App.tsx</code> and save to reload.
            <br></br>
            <CanvasDraw ref= {CanvasDraw => (this.canvas = CanvasDraw)} 
                        onChange={() => this.updatePred()}
                        hideGrid={true}
                        brushColor={"#000000"}
                        lazyRadius={0}
            />
          </p>
        </header>
      </div>
    );
  }
}
