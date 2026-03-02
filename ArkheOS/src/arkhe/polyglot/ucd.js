// ucd.js – Universal Coherence Detection in Node.js

function verifyConservation(C, F, tol = 1e-10) {
    return Math.abs(C + F - 1.0) < tol;
}

class UCD {
    constructor(data) {
        this.data = data;
        this.C = null;
        this.F = null;
    }

    analyze() {
        if (this.data.length > 1) {
            let sumCorr = 0;
            let n = this.data.length;
            for (let i = 0; i < n; i++) {
                for (let j = i + 1; j < n; j++) {
                    sumCorr += Math.abs(this._pearson(this.data[i], this.data[j]));
                }
            }
            let count = (n * (n - 1)) / 2;
            this.C = count > 0 ? sumCorr / count : 0.5;
        } else {
            this.C = 0.5;
        }
        this.F = 1.0 - this.C;
        return {
            C: this.C,
            F: this.F,
            conservation: verifyConservation(this.C, this.F),
            topology: this.C > 0.8 ? "toroidal" : "other"
        };
    }

    _pearson(x, y) {
        let n = x.length;
        let meanX = x.reduce((a, b) => a + b, 0) / n;
        let meanY = y.reduce((a, b) => a + b, 0) / n;
        let num = 0, denX = 0, denY = 0;
        for (let i = 0; i < n; i++) {
            let dx = x[i] - meanX;
            let dy = y[i] - meanY;
            num += dx * dy;
            denX += dx * dx;
            denY += dy * dy;
        }
        return (denX * denY === 0) ? 0 : num / Math.sqrt(denX * denY);
    }
}

const data = [[1, 2, 3, 4], [2, 3, 4, 5], [5, 6, 7, 8]];
const ucd = new UCD(data);
console.log(JSON.stringify(ucd.analyze(), null, 2));
