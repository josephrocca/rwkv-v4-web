const s = require("./sample");

describe('sample', () => {
    it('greedySampling', () => {
        expect(s.greedySampling([0,1])).toEqual(1);
        expect(s.greedySampling([2,1])).toEqual(0);
    });

    it('softmax', () => {
        const a = [-1, 0, 3, 5];
        const ans = [0.0021657, 0.00588697, 0.11824302, 0.87370431];

        s.softmax(a).forEach(
            (x, i) => expect(x).toBeCloseTo(ans[i])
        );
    });

    it('choiceIndex', () => {
        // choose first item with prob 0 and second with prob 1.  So should guarantee to give us 1
        expect(s.choiceIndex([0,1])).toEqual(1);

        // choose first item with prob 1 and second with prob 0.  So should guarantee to give us 0
        expect(s.choiceIndex([1,0])).toEqual(0);

        // choose first item with prob 1
        expect(s.choiceIndex([1])).toEqual(0);

        expect(s.choiceIndex([0,0,0,0,0,1,0,0,0,0])).toEqual(5);

        function expectedValue(arr) {
            var sum = 0;
            const N = 1000000;
            for(var i = 0; i < N; i++) {
                sum += s.choiceIndex(arr);
            }
            return sum / N;
        }

        // Test choiceIndex by checking the expected value
        expect(expectedValue([0,0.5, 0.5])).toBeCloseTo(1.5, 2);
        expect(expectedValue([0.25, 0.25, 0.5])).toBeCloseTo(0.25 + 0.5*2, 2); // 0* 0.25 + 1*0.25 + 2*0.5
    });

    it('getMultinomialProbs', () => {
        expect(s.getMultinomialProbs([-1, 0, 3, 5], 1, 0.8)).toEqual([0,0,0,1]); // The last element is so much larger than the others, it's guaranteed
        expect(s.getMultinomialProbs([5, 0, 3, 5], 1, 0.8)).toEqual([0.5,0,0,0.5]);
        expect(s.getMultinomialProbs([5, 5, 5, 5], 1, 0.8)).toEqual([0.25,0.25,0.25,0.25]);
        const a = [2.1, 2.5, 2.3, 2.2];
        const ans = [0.20753784191475178,0.3096100782635963,0.2534872925372871,0.22936478728436485];

        s.getMultinomialProbs(a, 1, 0.8).forEach(
            (x, i) => expect(x).toBeCloseTo(ans[i])
        );
    });

    it('getMultinomialProbs with temp 0.5', () => {
        // These first three, the temp should make no difference
        expect(s.getMultinomialProbs([-1, 0, 3, 5], 0.5, 0.8)).toEqual([0,0,0,1]); // The last element is so much larger than the others, it's guaranteed
        expect(s.getMultinomialProbs([5, 0, 3, 5], 0.5, 0.8)).toEqual([0.5,0,0,0.5]);
        expect(s.getMultinomialProbs([5, 5, 5, 5], 0.5, 0.8)).toEqual([0.25,0.25,0.25,0.25]);
        // Now with values close to each other.  Temperature will freeze out the lower two values
        const a = [2.1, 2.5, 2.3, 2.2];
        const ans = [0, 0.549833997312478, 0.45016600268752205, 0];

        console.log(s.getMultinomialProbs(a, 1, 0.5));
        s.getMultinomialProbs(a, 1, 0.5).forEach(
            (x, i) => expect(x).toBeCloseTo(ans[i])
        );
    });

    it('applyRepetitionPenalty with penalty of 1', () => {
        expect(s.applyRepetitionPenalty([-1, 0, 3, 5], [0], 1)).toEqual([-2, 0, 3, 5]);
        expect(s.applyRepetitionPenalty([-1, 0, 3, 5], [0, 2, 0], 1)).toEqual([-3, 0, 2, 5]);
        expect(s.applyRepetitionPenalty([-1, 0, 3, 5], [], 1)).toEqual([-1, 0, 3, 5]);
    });

    it('find_cutoff', () => {
        expect(s.find_cutoff([0.1, 0.25, 0.05, 0.6], 0.8)).toEqual(0.25);
        expect(s.find_cutoff([0.1, 0.25, 0.05, 0.6], 0.5)).toEqual(0.6);
        expect(s.find_cutoff([0.1, 0.25, 0.05, 0.6], 0.9)).toEqual(0.1);
        expect(s.find_cutoff([0.1, 0.25, 0.05, 0.6], 0.99)).toEqual(0.05);
        expect(s.find_cutoff([1], 0.99)).toEqual(1);
    })
});